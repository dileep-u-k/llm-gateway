const state = {
  uploadedArtifacts: [],
};

async function api(path, options = {}) {
  const token = document.getElementById('token')?.value?.trim();
  const headers = Object.assign({}, options.headers || {});
  if (token) headers.Authorization = `Bearer ${token}`;
  const tenant = document.getElementById('tenantId')?.value?.trim();
  const workspace = document.getElementById('workspaceId')?.value?.trim();
  if (tenant) headers['X-Tenant-ID'] = tenant;
  if (workspace) headers['X-Workspace-ID'] = workspace;
  const response = await fetch(path, { ...options, headers });
  const text = await response.text();
  const body = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(body.error || text || `Request failed with ${response.status}`);
  }
  return body;
}

function renderArtifacts(container, artifacts) {
  container.innerHTML = '';
  artifacts.forEach((artifact) => {
    const div = document.createElement('div');
    div.className = 'artifact-item';
    div.innerHTML = `<strong>${artifact.name || artifact.artifact_id}</strong>
      <div>${artifact.type || 'artifact'} · ${artifact.mime_type || 'unknown'}</div>
      <div>${artifact.artifact_id}</div>
      ${artifact.access_url ? `<a href="${artifact.access_url}" target="_blank" rel="noreferrer">Open artifact</a>` : ''}`;
    container.appendChild(div);
  });
}

function renderMetadata(response) {
  const panel = document.getElementById('metadataPanel');
  panel.innerHTML = '';
  const sections = [
    ['Route', response.route],
    ['Governance', response.governance],
    ['Security', response.security],
    ['Session', response.session],
    ['Generation', response.generation],
    ['Retrieval', response.retrieval],
  ];
  sections.forEach(([title, payload]) => {
    if (!payload) return;
    const div = document.createElement('div');
    div.className = 'meta-card';
    div.innerHTML = `<strong>${title}</strong><pre>${JSON.stringify(payload, null, 2)}</pre>`;
    panel.appendChild(div);
  });
}

function renderPlan(response) {
  const container = document.getElementById('executionPlan');
  container.innerHTML = '';
  const plan = response.execution_plan;
  if (!plan || !Array.isArray(plan.stages)) {
    container.innerHTML = '<div class="meta-card">No execution plan returned.</div>';
    return;
  }
  plan.stages.forEach((stage) => {
    const div = document.createElement('div');
    div.className = 'plan-stage';
    div.innerHTML = `<strong>${stage.title || stage.stage_id}</strong>
      <div>${stage.stage_type} · ${stage.capability}</div>
      <div>binding: ${stage.model_binding || 'dynamic'} · policy: ${stage.force_policy || 'dynamic'}</div>`;
    container.appendChild(div);
  });
}

function renderMedia(response) {
  const preview = document.getElementById('mediaPreview');
  preview.innerHTML = '';
  if (response.image_url) {
    preview.innerHTML += `<img src="${response.image_url}" alt="Generated output">`;
  }
  if (response.audio_url) {
    preview.innerHTML += `<audio controls src="${response.audio_url}"></audio>`;
  }
}

async function loadBootstrap() {
  const bootstrap = await api('/api/v1/platform/bootstrap');
  document.getElementById('authModeBadge').textContent = bootstrap.platform.auth_mode;
  document.getElementById('tenantId').value = bootstrap.platform.default_tenant;
  document.getElementById('workspaceId').value = bootstrap.platform.default_workspace;
  document.getElementById('conversationId').value ||= `session-${Date.now()}`;
  document.getElementById('prompt').value ||= bootstrap.sample_request.prompt;
  document.getElementById('simulationRequest');
}

async function uploadFiles() {
  const files = document.getElementById('fileInput').files;
  const token = document.getElementById('token').value.trim();
  const tenant = document.getElementById('tenantId').value.trim();
  const workspace = document.getElementById('workspaceId').value.trim();
  for (const file of files) {
    const form = new FormData();
    form.append('file', file);
    form.append('tenant_id', tenant);
    form.append('workspace_id', workspace);
    const headers = {};
    if (token) headers.Authorization = `Bearer ${token}`;
    const response = await fetch('/api/v1/assets/upload', { method: 'POST', headers, body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Upload failed');
    state.uploadedArtifacts.unshift(payload.artifact);
  }
  renderArtifacts(document.getElementById('uploadedArtifacts'), state.uploadedArtifacts);
}

async function loadArtifacts() {
  const payload = await api('/api/v1/artifacts');
  state.uploadedArtifacts = payload.artifacts || [];
  renderArtifacts(document.getElementById('uploadedArtifacts'), state.uploadedArtifacts);
}

async function sendRequest() {
  const responseStatus = document.getElementById('responseStatus');
  responseStatus.textContent = 'running';
  const body = {
    prompt: document.getElementById('prompt').value,
    user_id: document.getElementById('userId').value,
    tenant_id: document.getElementById('tenantId').value,
    workspace_id: document.getElementById('workspaceId').value,
    conversation_id: document.getElementById('conversationId').value,
    input_type: document.getElementById('inputType').value,
    task_type: document.getElementById('taskType').value,
    output_type: document.getElementById('outputType').value,
    requires_generation: document.getElementById('requiresGeneration').checked,
    requires_ocr: document.getElementById('requiresOCR').checked,
    requires_transcription: document.getElementById('requiresTranscription').checked,
    sync_or_async_preference: document.getElementById('syncPref').value,
    artifact_refs: state.uploadedArtifacts.map((artifact) => ({ artifact_id: artifact.artifact_id, role: 'input' })),
    config: {
      preference: document.getElementById('preference').value,
      answer_mode: document.getElementById('answerMode').value,
      force_model: document.getElementById('forceModel').value,
      force_scope: document.getElementById('forceScope').value,
      strict_force: document.getElementById('strictForce').checked,
    },
  };
  const payload = await api('/api/v1/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  const jobPanel = document.getElementById('jobPanel');
  jobPanel.innerHTML = '';
  if (payload.job) {
    responseStatus.textContent = 'async accepted';
    document.getElementById('responseContent').textContent = JSON.stringify(payload, null, 2);
    renderMetadata(payload);
    if (payload.job.async?.status_url) {
      jobPanel.innerHTML = `<div class="meta-card"><strong>Async Job</strong><div>${payload.job.job_id}</div><div>${payload.job.state}</div></div>`;
    }
    return;
  }

  responseStatus.textContent = payload.cache_status || 'completed';
  document.getElementById('responseContent').textContent = payload.content || JSON.stringify(payload, null, 2);
  renderMetadata(payload);
  renderPlan(payload);
  renderMedia(payload);
}

document.getElementById('loadBootstrap').addEventListener('click', () => loadBootstrap().catch(alert));
document.getElementById('uploadBtn').addEventListener('click', () => uploadFiles().catch(alert));
document.getElementById('loadArtifactsBtn').addEventListener('click', () => loadArtifacts().catch(alert));
document.getElementById('sendBtn').addEventListener('click', () => sendRequest().catch((err) => {
  document.getElementById('responseStatus').textContent = 'error';
  document.getElementById('responseContent').textContent = err.message;
}));

loadBootstrap().then(loadArtifacts).catch((err) => {
  document.getElementById('responseContent').textContent = err.message;
});
