async function adminApi(path, options = {}) {
  const token = document.getElementById('adminToken').value.trim();
  const headers = Object.assign({}, options.headers || {});
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(path, { ...options, headers });
  const text = await response.text();
  const body = text ? JSON.parse(text) : {};
  if (!response.ok) throw new Error(body.error || text || `Request failed with ${response.status}`);
  return body;
}

function renderList(target, items, formatter) {
  target.innerHTML = '';
  (items || []).forEach((item) => {
    const div = document.createElement('div');
    div.className = 'meta-card';
    div.innerHTML = formatter(item);
    target.appendChild(div);
  });
}

async function refreshOverview() {
  const bootstrap = await adminApi('/api/v1/platform/bootstrap');
  document.getElementById('adminAuthMode').textContent = bootstrap.platform.auth_mode;
  document.getElementById('simTenant').value ||= bootstrap.platform.default_tenant;
  document.getElementById('simWorkspace').value ||= bootstrap.platform.default_workspace;
  document.getElementById('overrideTenant').value ||= bootstrap.platform.default_tenant;
  document.getElementById('overrideWorkspace').value ||= bootstrap.platform.default_workspace;
  document.getElementById('simulationRequest').value ||= JSON.stringify(bootstrap.sample_request, null, 2);
  document.getElementById('overridePayload').value ||= JSON.stringify({
    name: 'custom',
    description: 'Example override',
    provider_allowlist: ['openai', 'anthropic'],
    capability_allowlist: ['text_generation', 'image_understanding'],
    force_scope_allowlist: ['primary_reasoner_force'],
    generation_allowed: true,
  }, null, 2);

  const overview = await adminApi('/api/v1/platform/admin/overview');
  renderList(document.getElementById('providerPanel'), overview.providers, (p) => `<strong>${p.name}</strong><div>${p.endpoint || 'endpoint unavailable'}</div><div>health: ${p.coarse_health}</div>`);
  renderList(document.getElementById('modelPanel'), overview.models, (m) => `<strong>${m.model_id}</strong><div>${m.provider} · ${m.quality_tier || 'n/a'}</div><div>${(m.capabilities || []).join(', ')}</div>`);
  renderList(document.getElementById('policyPanel'), overview.policies, (p) => `<strong>${p.tenant.name} / ${p.workspace.name}</strong><div>${p.policy.name}</div><div>${p.policy.description || ''}</div>`);
  renderList(document.getElementById('jobsPanel'), overview.recent_jobs, (j) => `<strong>${j.job_id}</strong><div>${j.state} · ${j.worker_class || 'worker'}</div><div>${j.error || ''}</div>`);
  renderList(document.getElementById('orchestrationsPanel'), overview.orchestrations, (o) => `<strong>${o.id || 'orchestration'}</strong><div>${o.prompt_preview || ''}</div><div>${o.model_used || ''}</div>`);
  renderList(document.getElementById('auditPanel'), overview.audit_events, (e) => `<strong>${e.action}</strong><div>${e.status} · ${e.summary || ''}</div><div>${e.tenant_id || ''}/${e.workspace_id || ''}</div>`);
  document.getElementById('metricsPanel').textContent = JSON.stringify(overview.metrics, null, 2);
}

async function simulatePolicy() {
  const payload = JSON.parse(document.getElementById('simulationRequest').value);
  payload.tenant_id = document.getElementById('simTenant').value.trim();
  payload.workspace_id = document.getElementById('simWorkspace').value.trim();
  const result = await adminApi('/api/v1/platform/admin/policies/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  document.getElementById('simulationOutput').textContent = JSON.stringify(result, null, 2);
}

async function saveOverride() {
  const tenant = document.getElementById('overrideTenant').value.trim();
  const workspace = document.getElementById('overrideWorkspace').value.trim();
  const payload = JSON.parse(document.getElementById('overridePayload').value);
  const result = await adminApi(`/api/v1/platform/admin/policies/${tenant}/${workspace}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  document.getElementById('simulationOutput').textContent = JSON.stringify(result, null, 2);
  await refreshOverview();
}

document.getElementById('refreshOverview').addEventListener('click', () => refreshOverview().catch(alert));
document.getElementById('simulatePolicy').addEventListener('click', () => simulatePolicy().catch(alert));
document.getElementById('saveOverride').addEventListener('click', () => saveOverride().catch(alert));

refreshOverview().catch((err) => {
  document.getElementById('simulationOutput').textContent = err.message;
});
