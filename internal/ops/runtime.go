package ops

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/dileep-u-k/llm-gateway/internal/api"
	"github.com/dileep-u-k/llm-gateway/internal/llm"
	"github.com/dileep-u-k/llm-gateway/internal/observability"
	"github.com/redis/go-redis/v9"
)

const (
	jobKeyPrefix        = "async:job:"
	jobQueueKey         = "async:job_queue"
	jobIndexKey         = "async:job_index"
	deadLetterQueueKey  = "async:dead_letter"
	checkpointKeyPrefix = "async:checkpoint:"
)

type JobState string

const (
	JobStateQueued             JobState = "queued"
	JobStateRunning            JobState = "running"
	JobStateCheckpointed       JobState = "checkpointed"
	JobStateRetrying           JobState = "retrying"
	JobStateCompleted          JobState = "completed"
	JobStateFailed             JobState = "failed"
	JobStateCancelled          JobState = "cancelled"
	JobStatePartiallyCompleted JobState = "partially_completed"
)

type FailureClass string

const (
	FailureProviderTimeout FailureClass = "provider_timeout"
	FailureProvider5xx     FailureClass = "provider_5xx"
	FailureRateLimit       FailureClass = "rate_limit"
	FailureMalformedOutput FailureClass = "malformed_output"
	FailureVectorDB        FailureClass = "vector_db_failure"
	FailureTool            FailureClass = "tool_failure"
	FailureWorkerCrash     FailureClass = "worker_crash"
	FailureArtifactLoad    FailureClass = "artifact_load_failure"
	FailureGeneration      FailureClass = "generation_failure"
	FailureCheckpoint      FailureClass = "checkpoint_failure"
	FailurePolicy          FailureClass = "policy_validation_failure"
	FailureUnknown         FailureClass = "unknown"
)

type JobCheckpoint struct {
	StageID      string            `json:"stage_id,omitempty"`
	StageType    string            `json:"stage_type,omitempty"`
	WorkerClass  string            `json:"worker_class,omitempty"`
	Status       string            `json:"status,omitempty"`
	Attempts     int               `json:"attempts,omitempty"`
	StartedAt    time.Time         `json:"started_at,omitempty"`
	CompletedAt  time.Time         `json:"completed_at,omitempty"`
	FailureClass FailureClass      `json:"failure_class,omitempty"`
	Error        string            `json:"error,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

type JobRecord struct {
	JobID           string                  `json:"job_id,omitempty"`
	State           JobState                `json:"state,omitempty"`
	WorkerClass     string                  `json:"worker_class,omitempty"`
	Request         api.GenerationRequest   `json:"request"`
	Prepared        *llm.PreparedExecution  `json:"prepared,omitempty"`
	Checkpoints     []JobCheckpoint         `json:"checkpoints,omitempty"`
	Result          *api.GenerationResponse `json:"result,omitempty"`
	PartialResponse *api.GenerationResponse `json:"partial_response,omitempty"`
	FailureClass    FailureClass            `json:"failure_class,omitempty"`
	Error           string                  `json:"error,omitempty"`
	RetryCount      int                     `json:"retry_count,omitempty"`
	CancelRequested bool                    `json:"cancel_requested,omitempty"`
	DeadLettered    bool                    `json:"dead_lettered,omitempty"`
	TraceID         string                  `json:"trace_id,omitempty"`
	CreatedAt       time.Time               `json:"created_at,omitempty"`
	StartedAt       time.Time               `json:"started_at,omitempty"`
	CompletedAt     time.Time               `json:"completed_at,omitempty"`
	UpdatedAt       time.Time               `json:"updated_at,omitempty"`
}

type Config struct {
	Workers               int
	QueuePollTimeout      time.Duration
	JobTTL                time.Duration
	MaxRetries            int
	BackpressureThreshold int64
	CallbackTimeout       time.Duration
}

func DefaultConfig() Config {
	return Config{
		Workers:               3,
		QueuePollTimeout:      2 * time.Second,
		JobTTL:                72 * time.Hour,
		MaxRetries:            3,
		BackpressureThreshold: 25,
		CallbackTimeout:       5 * time.Second,
	}
}

type PrepareFunc func(context.Context, api.GenerationRequest) (*llm.PreparedExecution, error)

func (f PrepareFunc) Prepare(ctx context.Context, req api.GenerationRequest) (*llm.PreparedExecution, error) {
	return f(ctx, req)
}

type ExecuteFunc func(context.Context, api.GenerationRequest, *llm.PreparedExecution) (api.GenerationResponse, error)

func (f ExecuteFunc) Execute(ctx context.Context, req api.GenerationRequest, prepared *llm.PreparedExecution) (api.GenerationResponse, error) {
	return f(ctx, req, prepared)
}

type Preparer interface {
	Prepare(context.Context, api.GenerationRequest) (*llm.PreparedExecution, error)
}

type Executor interface {
	Execute(context.Context, api.GenerationRequest, *llm.PreparedExecution) (api.GenerationResponse, error)
}

type Runtime struct {
	rdb       *redis.Client
	store     *jobStore
	queue     *jobQueue
	preparer  Preparer
	executor  Executor
	retry     RetryManager
	config    Config
	client    *http.Client
	obs       *observability.Registry
	startOnce sync.Once
}

func NewRuntime(rdb *redis.Client, preparer Preparer, executor Executor, config Config, obs *observability.Registry) *Runtime {
	if config.Workers <= 0 {
		config = DefaultConfig()
	}
	if config.QueuePollTimeout <= 0 {
		config.QueuePollTimeout = 2 * time.Second
	}
	if config.JobTTL <= 0 {
		config.JobTTL = 72 * time.Hour
	}
	if config.MaxRetries <= 0 {
		config.MaxRetries = 3
	}
	if config.CallbackTimeout <= 0 {
		config.CallbackTimeout = 5 * time.Second
	}
	if obs == nil {
		obs = observability.Default()
	}
	return &Runtime{
		rdb:      rdb,
		store:    newJobStore(rdb, config.JobTTL),
		queue:    newJobQueue(rdb),
		preparer: preparer,
		executor: executor,
		retry: RetryManager{
			MaxRetries: config.MaxRetries,
			BaseDelay:  500 * time.Millisecond,
			MaxDelay:   8 * time.Second,
		},
		config: config,
		client: &http.Client{Timeout: config.CallbackTimeout},
		obs:    obs,
	}
}

func (r *Runtime) Start(ctx context.Context) {
	r.startOnce.Do(func() {
		for i := 0; i < r.config.Workers; i++ {
			go r.workerLoop(ctx, i+1)
		}
	})
}

func (r *Runtime) Submit(ctx context.Context, req api.GenerationRequest, prepared *llm.PreparedExecution) (*JobRecord, error) {
	if prepared == nil && r.preparer != nil {
		nextPrepared, err := r.preparer.Prepare(ctx, req)
		if err != nil {
			return nil, err
		}
		prepared = nextPrepared
	}

	now := time.Now().UTC()
	job := &JobRecord{
		JobID:       llm.GenerateCacheKey(strings.Join([]string{req.UserID, req.ConversationID, req.Prompt, now.Format(time.RFC3339Nano)}, "|")),
		State:       JobStateQueued,
		WorkerClass: workerClassFromPrepared(prepared),
		Request:     req,
		Prepared:    prepared,
		Checkpoints: checkpointsFromPrepared(prepared),
		CreatedAt:   now,
		UpdatedAt:   now,
	}
	job.TraceID = r.obs.StartTrace("async_job_submission", job.JobID)
	r.obs.RecordJobState(string(job.State))
	r.obs.RecordLog("info", "async", "accepted async job", map[string]string{
		"job_id":       job.JobID,
		"worker_class": job.WorkerClass,
	})
	if err := r.store.Save(ctx, job); err != nil {
		return nil, err
	}
	if err := r.queue.Enqueue(ctx, job.JobID); err != nil {
		return nil, err
	}
	if depth, err := r.queue.Len(ctx); err == nil && depth > r.config.BackpressureThreshold {
		r.obs.IncCounter("queue_backpressure_events")
		r.obs.RecordLog("warn", "async", "job queue backpressure threshold exceeded", map[string]string{
			"depth": fmt.Sprintf("%d", depth),
		})
	}
	return job, nil
}

func (r *Runtime) GetJob(ctx context.Context, jobID string) (*JobRecord, error) {
	return r.store.Get(ctx, jobID)
}

func (r *Runtime) CancelJob(ctx context.Context, jobID string) error {
	job, err := r.store.Get(ctx, jobID)
	if err != nil {
		return err
	}
	if job == nil {
		return fmt.Errorf("job %s not found", jobID)
	}
	if job.State == JobStateCompleted || job.State == JobStateFailed || job.State == JobStateCancelled {
		return nil
	}
	job.CancelRequested = true
	if job.State == JobStateQueued || job.State == JobStateRetrying {
		job.State = JobStateCancelled
		job.CompletedAt = time.Now().UTC()
	}
	job.UpdatedAt = time.Now().UTC()
	r.obs.RecordJobState(string(job.State))
	r.obs.RecordLog("info", "async", "cancel requested for job", map[string]string{"job_id": job.JobID})
	return r.store.Save(ctx, job)
}

func (r *Runtime) workerLoop(ctx context.Context, workerNumber int) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		jobID, err := r.queue.Dequeue(ctx, r.config.QueuePollTimeout)
		if err != nil {
			if err == context.Canceled || err == context.DeadlineExceeded {
				return
			}
			if err == redis.Nil {
				continue
			}
			r.obs.RecordLog("error", "async", "worker dequeue failed", map[string]string{"error": err.Error()})
			continue
		}
		r.processJob(ctx, jobID, workerNumber)
	}
}

func (r *Runtime) processJob(ctx context.Context, jobID string, workerNumber int) {
	job, err := r.store.Get(ctx, jobID)
	if err != nil || job == nil {
		r.obs.RecordLog("error", "async", "failed to load job from store", map[string]string{
			"job_id": jobID,
			"error":  errorString(err),
		})
		return
	}
	if job.State == JobStateCompleted || job.State == JobStateCancelled {
		return
	}
	if job.CancelRequested {
		job.State = JobStateCancelled
		job.CompletedAt = time.Now().UTC()
		job.UpdatedAt = time.Now().UTC()
		_ = r.store.Save(ctx, job)
		r.obs.RecordJobState(string(job.State))
		r.obs.FinishTrace(job.TraceID, string(job.State))
		return
	}
	if job.Prepared == nil && r.preparer != nil {
		prepared, err := r.preparer.Prepare(ctx, job.Request)
		if err != nil {
			r.failJob(ctx, job, FailureCheckpoint, err, nil)
			return
		}
		job.Prepared = prepared
		job.WorkerClass = workerClassFromPrepared(prepared)
		job.Checkpoints = checkpointsFromPrepared(prepared)
	}

	startedAt := time.Now().UTC()
	if job.StartedAt.IsZero() {
		job.StartedAt = startedAt
	}
	job.State = JobStateRunning
	job.UpdatedAt = startedAt
	_ = r.store.Save(ctx, job)
	r.obs.RecordJobState(string(job.State))
	r.obs.RecordTraceSpan(job.TraceID, "job.start", "running", map[string]string{
		"job_id":       job.JobID,
		"worker_class": job.WorkerClass,
		"worker_id":    fmt.Sprintf("%d", workerNumber),
	})

	if len(job.Checkpoints) == 0 {
		job.Checkpoints = checkpointsFromPrepared(job.Prepared)
	}

	for idx := range job.Checkpoints {
		checkpoint := &job.Checkpoints[idx]
		if checkpoint.Status == "completed" || checkpoint.Status == "skipped" {
			continue
		}
		if job.CancelRequested {
			checkpoint.Status = "cancelled"
			checkpoint.CompletedAt = time.Now().UTC()
			job.State = JobStateCancelled
			job.CompletedAt = checkpoint.CompletedAt
			job.UpdatedAt = checkpoint.CompletedAt
			_ = r.store.Save(ctx, job)
			r.obs.RecordJobState(string(job.State))
			r.obs.FinishTrace(job.TraceID, string(job.State))
			return
		}

		checkpoint.WorkerClass = stageWorkerClass(checkpoint.StageType)
		checkpoint.Status = "running"
		checkpoint.Attempts++
		checkpoint.StartedAt = time.Now().UTC()
		r.obs.RecordStage(checkpoint.StageType, checkpoint.Status)
		r.obs.RecordTraceSpan(job.TraceID, checkpoint.StageID, "running", map[string]string{
			"stage_type":   checkpoint.StageType,
			"worker_class": checkpoint.WorkerClass,
		})
		_ = r.store.SaveCheckpoint(ctx, job.JobID, *checkpoint)
		_ = r.store.Save(ctx, job)

		if isPrimaryCheckpoint(job.Prepared, checkpoint.StageID) {
			resp, execErr := r.executor.Execute(ctx, job.Request, job.Prepared)
			if execErr != nil {
				r.handleCheckpointFailure(ctx, job, checkpoint, execErr)
				return
			}
			job.Result = &resp
			checkpoint.Metadata = mergeMetadata(checkpoint.Metadata, map[string]string{
				"model_used":       resp.ModelUsed,
				"orchestration_id": routeOrchestrationID(resp.Route),
			})
			r.observeResponse(job.Request, job.Prepared, resp)
		}

		if job.Result != nil {
			checkpoint.Metadata = mergeMetadata(checkpoint.Metadata, map[string]string{
				"state": string(job.State),
			})
		}
		checkpoint.Status = "completed"
		checkpoint.Error = ""
		checkpoint.CompletedAt = time.Now().UTC()
		job.State = JobStateCheckpointed
		job.UpdatedAt = checkpoint.CompletedAt
		_ = r.store.SaveCheckpoint(ctx, job.JobID, *checkpoint)
		r.obs.RecordStage(checkpoint.StageType, checkpoint.Status)
	}

	job.State = JobStateCompleted
	job.CompletedAt = time.Now().UTC()
	job.UpdatedAt = job.CompletedAt
	if err := r.store.Save(ctx, job); err != nil {
		r.failJob(ctx, job, FailureCheckpoint, err, nil)
		return
	}
	r.obs.RecordJobState(string(job.State))
	r.obs.ObserveJobDuration(job.CompletedAt.Sub(job.StartedAt))
	r.obs.RecordTraceSpan(job.TraceID, "job.complete", "completed", map[string]string{"job_id": job.JobID})
	r.obs.FinishTrace(job.TraceID, string(job.State))
	r.sendCallback(ctx, job)
}

func (r *Runtime) handleCheckpointFailure(ctx context.Context, job *JobRecord, checkpoint *JobCheckpoint, err error) {
	failureClass := r.retry.Classify(err)
	checkpoint.Status = "failed"
	checkpoint.Error = err.Error()
	checkpoint.FailureClass = failureClass
	checkpoint.CompletedAt = time.Now().UTC()
	job.FailureClass = failureClass
	job.Error = err.Error()
	job.UpdatedAt = checkpoint.CompletedAt
	_ = r.store.SaveCheckpoint(ctx, job.JobID, *checkpoint)

	if degradedReq, degradedPrepared, ok := degradedFallback(job.Request, job.Prepared, failureClass); ok {
		job.Request = degradedReq
		job.Prepared = degradedPrepared
		r.obs.IncCounter("degraded_mode_activations")
		r.obs.RecordLog("warn", "async", "switching job to degraded mode", map[string]string{
			"job_id":        job.JobID,
			"failure_class": string(failureClass),
		})
	}

	if r.retry.ShouldRetry(failureClass, checkpoint.Attempts) {
		job.RetryCount++
		job.State = JobStateRetrying
		checkpoint.Status = "retrying"
		delay := r.retry.NextDelay(checkpoint.Attempts, failureClass)
		r.obs.IncCounter("async_retry_" + string(failureClass))
		r.obs.RecordJobState(string(job.State))
		_ = r.store.Save(ctx, job)
		go r.enqueueAfter(job.JobID, delay)
		return
	}

	r.failJob(ctx, job, failureClass, err, checkpoint)
}

func (r *Runtime) failJob(ctx context.Context, job *JobRecord, failureClass FailureClass, err error, checkpoint *JobCheckpoint) {
	now := time.Now().UTC()
	job.FailureClass = failureClass
	job.Error = errorString(err)
	job.UpdatedAt = now
	job.CompletedAt = now
	if job.PartialResponse != nil || job.Result != nil {
		job.State = JobStatePartiallyCompleted
	} else {
		job.State = JobStateFailed
	}
	if checkpoint != nil && checkpoint.Status == "" {
		checkpoint.Status = "failed"
		checkpoint.CompletedAt = now
		checkpoint.Error = errorString(err)
		checkpoint.FailureClass = failureClass
	}
	if job.RetryCount >= r.config.MaxRetries {
		job.DeadLettered = true
	}
	_ = r.store.Save(ctx, job)
	if job.DeadLettered {
		_ = r.store.PushDeadLetter(ctx, job.JobID)
	}
	r.obs.RecordJobState(string(job.State))
	r.obs.RecordLog("error", "async", "job failed", map[string]string{
		"job_id":        job.JobID,
		"failure_class": string(failureClass),
		"error":         errorString(err),
	})
	r.obs.FinishTrace(job.TraceID, string(job.State))
	r.sendCallback(ctx, job)
}

func (r *Runtime) enqueueAfter(jobID string, delay time.Duration) {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	<-timer.C
	_ = r.queue.Enqueue(context.Background(), jobID)
}

func (r *Runtime) observeResponse(req api.GenerationRequest, prepared *llm.PreparedExecution, resp api.GenerationResponse) {
	modalities := []string{"text"}
	taskType := "general"
	if prepared != nil {
		if len(prepared.Task.Modalities) > 0 {
			modalities = prepared.Task.Modalities
		}
		taskType = firstNonEmpty(prepared.Task.TaskType, taskType)
	}
	for _, modality := range modalities {
		r.obs.RecordModality(modality)
	}
	forceScope := ""
	if resp.Route != nil {
		forceScope = resp.Route.ForcedSemantics.Scope
		r.obs.RecordRoute(resp.Route.SelectedProvider, resp.Route.SelectedModel, resp.Route.Strategy, resp.Route.RouteFamily, forceScope)
	}
	cost := llm.EstimateRequestCost(resp.ModelUsed, resp.Usage, len(resp.GeneratedArtifacts))
	r.obs.ObserveCost(llm.ProviderForModel(resp.ModelUsed), taskType, strings.Join(modalities, "+"), cost)
}

func (r *Runtime) sendCallback(ctx context.Context, job *JobRecord) {
	if job == nil || strings.TrimSpace(job.Request.CallbackURL) == "" {
		return
	}
	status := job.StatusResponse()
	payload, err := json.Marshal(map[string]any{
		"job":    status,
		"result": job.Result,
	})
	if err != nil {
		r.obs.RecordLog("error", "async", "failed to marshal callback payload", map[string]string{"job_id": job.JobID})
		return
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, job.Request.CallbackURL, bytes.NewReader(payload))
	if err != nil {
		r.obs.RecordLog("error", "async", "failed to create callback request", map[string]string{"job_id": job.JobID, "error": err.Error()})
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if _, err := r.client.Do(req); err != nil {
		r.obs.RecordLog("error", "async", "callback delivery failed", map[string]string{"job_id": job.JobID, "error": err.Error()})
	}
}

func (j *JobRecord) AcceptedResponse() *api.JobAcceptedResponse {
	if j == nil {
		return nil
	}
	return &api.JobAcceptedResponse{
		JobID:         j.JobID,
		State:         string(j.State),
		AcceptedAt:    formatTime(j.CreatedAt),
		ExecutionPlan: buildExecutionPlanMetadata(j.Prepared),
		Async: &api.AsyncMetadata{
			Accepted:    true,
			JobID:       j.JobID,
			State:       string(j.State),
			StatusURL:   fmt.Sprintf("/api/v1/jobs/%s", j.JobID),
			ResultURL:   fmt.Sprintf("/api/v1/jobs/%s/result", j.JobID),
			CancelURL:   fmt.Sprintf("/api/v1/jobs/%s/cancel", j.JobID),
			PollAfterMS: 1000,
		},
	}
}

func (j *JobRecord) StatusResponse() *api.JobStatusResponse {
	if j == nil {
		return nil
	}
	status := &api.JobStatusResponse{
		JobID:            j.JobID,
		State:            string(j.State),
		WorkerClass:      j.WorkerClass,
		FailureClass:     string(j.FailureClass),
		Error:            j.Error,
		RetryCount:       j.RetryCount,
		DeadLettered:     j.DeadLettered,
		ResultAvailable:  j.Result != nil,
		CreatedAt:        formatTime(j.CreatedAt),
		StartedAt:        formatTime(j.StartedAt),
		CompletedAt:      formatTime(j.CompletedAt),
		UpdatedAt:        formatTime(j.UpdatedAt),
		ExecutionPlan:    buildExecutionPlanMetadata(j.Prepared),
		PartialResponse:  firstResponse(j.PartialResponse, j.Result),
		GeneratedTraceID: j.TraceID,
		Async: &api.AsyncMetadata{
			Accepted:    true,
			JobID:       j.JobID,
			State:       string(j.State),
			StatusURL:   fmt.Sprintf("/api/v1/jobs/%s", j.JobID),
			ResultURL:   fmt.Sprintf("/api/v1/jobs/%s/result", j.JobID),
			CancelURL:   fmt.Sprintf("/api/v1/jobs/%s/cancel", j.JobID),
			PollAfterMS: 1000,
		},
	}
	for _, checkpoint := range j.Checkpoints {
		status.Checkpoints = append(status.Checkpoints, api.JobCheckpointMetadata{
			StageID:      checkpoint.StageID,
			StageType:    checkpoint.StageType,
			WorkerClass:  checkpoint.WorkerClass,
			Status:       checkpoint.Status,
			Attempts:     checkpoint.Attempts,
			StartedAt:    formatTime(checkpoint.StartedAt),
			CompletedAt:  formatTime(checkpoint.CompletedAt),
			FailureClass: string(checkpoint.FailureClass),
			Error:        checkpoint.Error,
			Metadata:     cloneStringMap(checkpoint.Metadata),
		})
	}
	return status
}

type jobStore struct {
	rdb *redis.Client
	ttl time.Duration
}

func newJobStore(rdb *redis.Client, ttl time.Duration) *jobStore {
	return &jobStore{rdb: rdb, ttl: ttl}
}

func (s *jobStore) Save(ctx context.Context, job *JobRecord) error {
	if job == nil {
		return nil
	}
	payload, err := json.Marshal(job)
	if err != nil {
		return err
	}
	pipe := s.rdb.TxPipeline()
	pipe.Set(ctx, jobKeyPrefix+job.JobID, payload, s.ttl)
	pipe.LPush(ctx, jobIndexKey, job.JobID)
	pipe.LTrim(ctx, jobIndexKey, 0, 199)
	pipe.Expire(ctx, jobIndexKey, s.ttl)
	_, err = pipe.Exec(ctx)
	return err
}

func (s *jobStore) Get(ctx context.Context, jobID string) (*JobRecord, error) {
	payload, err := s.rdb.Get(ctx, jobKeyPrefix+jobID).Bytes()
	if err == redis.Nil {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var job JobRecord
	if err := json.Unmarshal(payload, &job); err != nil {
		return nil, err
	}
	return &job, nil
}

func (s *jobStore) SaveCheckpoint(ctx context.Context, jobID string, checkpoint JobCheckpoint) error {
	payload, err := json.Marshal(checkpoint)
	if err != nil {
		return err
	}
	return s.rdb.Set(ctx, checkpointStoreKey(jobID, checkpoint.StageID), payload, s.ttl).Err()
}

func (s *jobStore) PushDeadLetter(ctx context.Context, jobID string) error {
	pipe := s.rdb.TxPipeline()
	pipe.LPush(ctx, deadLetterQueueKey, jobID)
	pipe.LTrim(ctx, deadLetterQueueKey, 0, 199)
	pipe.Expire(ctx, deadLetterQueueKey, s.ttl)
	_, err := pipe.Exec(ctx)
	return err
}

type jobQueue struct {
	rdb *redis.Client
}

func newJobQueue(rdb *redis.Client) *jobQueue {
	return &jobQueue{rdb: rdb}
}

func (q *jobQueue) Enqueue(ctx context.Context, jobID string) error {
	return q.rdb.LPush(ctx, jobQueueKey, jobID).Err()
}

func (q *jobQueue) Dequeue(ctx context.Context, timeout time.Duration) (string, error) {
	values, err := q.rdb.BRPop(ctx, timeout, jobQueueKey).Result()
	if err != nil {
		return "", err
	}
	if len(values) < 2 {
		return "", redis.Nil
	}
	return values[1], nil
}

func (q *jobQueue) Len(ctx context.Context) (int64, error) {
	return q.rdb.LLen(ctx, jobQueueKey).Result()
}

type RetryManager struct {
	MaxRetries int
	BaseDelay  time.Duration
	MaxDelay   time.Duration
}

func (m RetryManager) Classify(err error) FailureClass {
	if err == nil {
		return ""
	}
	message := strings.ToLower(err.Error())
	switch {
	case strings.Contains(message, "timeout"), strings.Contains(message, "deadline"):
		return FailureProviderTimeout
	case strings.Contains(message, "429"), strings.Contains(message, "rate limit"):
		return FailureRateLimit
	case strings.Contains(message, "5xx"), strings.Contains(message, "502"), strings.Contains(message, "503"), strings.Contains(message, "504"):
		return FailureProvider5xx
	case strings.Contains(message, "vector"), strings.Contains(message, "rag retrieval failed"):
		return FailureVectorDB
	case strings.Contains(message, "tool"):
		return FailureTool
	case strings.Contains(message, "artifact"):
		return FailureArtifactLoad
	case strings.Contains(message, "checkpoint"):
		return FailureCheckpoint
	case strings.Contains(message, "policy"):
		return FailurePolicy
	case strings.Contains(message, "generation"), strings.Contains(message, "image"), strings.Contains(message, "speech"):
		return FailureGeneration
	default:
		return FailureUnknown
	}
}

func (m RetryManager) ShouldRetry(class FailureClass, attempts int) bool {
	if attempts >= m.MaxRetries {
		return false
	}
	switch class {
	case FailureProviderTimeout, FailureProvider5xx, FailureRateLimit, FailureVectorDB, FailureTool, FailureGeneration, FailureUnknown:
		return true
	default:
		return false
	}
}

func (m RetryManager) NextDelay(attempts int, class FailureClass) time.Duration {
	delay := m.BaseDelay
	for i := 1; i < attempts; i++ {
		delay *= 2
		if delay >= m.MaxDelay {
			delay = m.MaxDelay
			break
		}
	}
	if class == FailureRateLimit && delay < 2*time.Second {
		delay = 2 * time.Second
	}
	return delay
}

func degradedFallback(req api.GenerationRequest, prepared *llm.PreparedExecution, class FailureClass) (api.GenerationRequest, *llm.PreparedExecution, bool) {
	nextReq := req
	var nextPrepared *llm.PreparedExecution
	if prepared != nil {
		copyPrepared := *prepared
		nextPrepared = &copyPrepared
		copyTask := prepared.Task
		nextPrepared.Task = copyTask
		if prepared.Plan != nil {
			copyPlan := *prepared.Plan
			copyPlan.Stages = append([]llm.ExecutionStage(nil), prepared.Plan.Stages...)
			nextPrepared.Plan = &copyPlan
		}
	}
	switch class {
	case FailureVectorDB:
		nextReq.Config.AnswerMode = "best_effort"
		if nextPrepared != nil {
			nextPrepared.Task.NeedsRetrieval = false
		}
		return nextReq, nextPrepared, true
	case FailureTool:
		if nextPrepared != nil {
			nextPrepared.Task.NeedsTooling = false
		}
		return nextReq, nextPrepared, true
	default:
		return req, prepared, false
	}
}

func workerClassFromPrepared(prepared *llm.PreparedExecution) string {
	if prepared == nil {
		return "reasoning_worker"
	}
	switch prepared.Task.PrimaryCapability {
	case llm.CapabilityEmbeddings:
		return "retrieval_worker"
	case llm.CapabilityOCR:
		return "ocr_worker"
	case llm.CapabilityTranscription:
		return "transcription_worker"
	case llm.CapabilityImageUnderstanding:
		return "vision_worker"
	case llm.CapabilityImageGeneration, llm.CapabilityImageEditing, llm.CapabilityVideoGeneration:
		return "generation_worker"
	case llm.CapabilityTTS:
		return "synthesis_worker"
	default:
		return "reasoning_worker"
	}
}

func stageWorkerClass(stageType string) string {
	switch stageType {
	case "retrieval":
		return "retrieval_worker"
	case "ocr":
		return "ocr_worker"
	case "transcription":
		return "transcription_worker"
	case "video_understanding":
		return "vision_worker"
	case "image_generation", "image_editing", "storyboard_generation", "asset_bundle":
		return "generation_worker"
	case "speech_synthesis":
		return "synthesis_worker"
	case "audio_post_processing", "quality_validation", "artifact_storage":
		return "post_processing_worker"
	case "evaluation":
		return "evaluation_worker"
	default:
		return "reasoning_worker"
	}
}

func checkpointsFromPrepared(prepared *llm.PreparedExecution) []JobCheckpoint {
	if prepared == nil || prepared.Plan == nil || len(prepared.Plan.Stages) == 0 {
		return nil
	}
	checkpoints := make([]JobCheckpoint, 0, len(prepared.Plan.Stages))
	for _, stage := range prepared.Plan.Stages {
		checkpoints = append(checkpoints, JobCheckpoint{
			StageID:     stage.StageID,
			StageType:   stage.StageType,
			WorkerClass: stageWorkerClass(stage.StageType),
			Status:      "planned",
		})
	}
	return checkpoints
}

func isPrimaryCheckpoint(prepared *llm.PreparedExecution, stageID string) bool {
	return prepared != nil && prepared.Plan != nil && prepared.Plan.PrimaryStageID == stageID
}

func checkpointStoreKey(jobID, stageID string) string {
	return checkpointKeyPrefix + jobID + ":" + stageID
}

func routeOrchestrationID(route *api.RouteMetadata) string {
	if route == nil {
		return ""
	}
	return route.OrchestrationID
}

func buildExecutionPlanMetadata(prepared *llm.PreparedExecution) *api.ExecutionPlanMetadata {
	if prepared == nil || prepared.Plan == nil {
		return nil
	}
	plan := prepared.Plan
	out := &api.ExecutionPlanMetadata{
		PlanID:            plan.PlanID,
		PlanType:          plan.PlanType,
		SyncMode:          plan.SyncMode,
		CostTier:          plan.CostTier,
		LatencyTier:       plan.LatencyTier,
		PrimaryStageID:    plan.PrimaryStageID,
		PrimaryCapability: string(plan.PrimaryCapability),
		ForceScope:        plan.ForceScope,
		RequiresAsync:     plan.RequiresAsync,
		Modalities:        append([]string(nil), plan.Modalities...),
		Notes:             append([]string(nil), plan.Notes...),
	}
	for _, capability := range plan.RequiredCapabilities {
		out.RequiredCapabilities = append(out.RequiredCapabilities, string(capability))
	}
	for _, stage := range plan.Stages {
		out.Stages = append(out.Stages, api.ExecutionStageMetadata{
			StageID:          stage.StageID,
			StageType:        stage.StageType,
			Title:            stage.Title,
			Capability:       string(stage.Capability),
			ModelBinding:     stage.ModelBinding,
			DependsOn:        append([]string(nil), stage.DependsOn...),
			ForcePolicy:      stage.ForcePolicy,
			BindingViolation: stage.BindingViolation,
			Status:           stage.Status,
			Optional:         stage.Optional,
			ForceApplied:     stage.ForceApplied,
			Strict:           stage.Strict,
		})
	}
	return out
}

func firstResponse(values ...*api.GenerationResponse) *api.GenerationResponse {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func formatTime(ts time.Time) string {
	if ts.IsZero() {
		return ""
	}
	return ts.UTC().Format(time.RFC3339Nano)
}

func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func mergeMetadata(base, extra map[string]string) map[string]string {
	if len(base) == 0 && len(extra) == 0 {
		return nil
	}
	out := cloneStringMap(base)
	if out == nil {
		out = make(map[string]string)
	}
	for key, value := range extra {
		if value != "" {
			out[key] = value
		}
	}
	return out
}
