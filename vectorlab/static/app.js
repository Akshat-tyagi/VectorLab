'use strict';

const $ = (id) => document.getElementById(id);
let documents = [];
let page = 0;
let loadedDocuments = false;
const pageSize = 10;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function notice(message, error = false) {
  $('notice').textContent = message;
  $('notice').className = error ? 'notice error' : 'notice';
  $('notice').hidden = !message;
}

async function request(path, options = {}) {
  let response;
  try { response = await fetch(path, { cache: 'no-store', ...options }); }
  catch { throw new Error('Cannot reach VectorLab. Make sure the FastAPI server is running, then refresh the page.'); }
  let data;
  try { data = await response.json(); }
  catch { throw new Error('The server returned an unexpected response. Check that the backend is running.'); }
  if (!response.ok) {
    let detail = data.detail || data.error || 'Request failed.';
    if (Array.isArray(detail)) detail = detail.map((item) => item.msg).join('; ');
    if (response.status >= 500 && response.status !== 503) detail = 'The server could not complete this request. Check the embedding service and server logs.';
    throw new Error(String(detail));
  }
  return data;
}

function post(path, data) {
  return request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
}

async function refreshStatus() {
  try {
    const status = await request('/status');
    $('count').textContent = status.documents.toLocaleString();
    $('dimensions').textContent = status.dims ?? 'Not set';
    $('model').textContent = status.model.replace('sentence-transformers/', '');
    $('model').title = status.model;
    $('generation-help').textContent = status.generation_configured
      ? `Answers use Groq · ${status.generation_model}. Your question and retrieved passages are sent to Groq.`
      : 'Add GROQ_API_KEY to the server .env file and restart to enable Groq answers. Search works without a key.';
    $('connection').textContent = 'Backend connected';
    $('connection').className = 'connection online';
  } catch (error) {
    $('connection').textContent = 'Backend unavailable';
    $('connection').className = 'connection offline';
    notice(error.message, true);
  }
}

function empty(container, title, text) {
  const box = element('div', 'empty-state');
  box.append(element('h3', '', title), element('p', '', text));
  container.replaceChildren(box);
}

function passageCard(context, rank, deletable = false) {
  const card = element('article', 'result-card');
  const top = element('div', 'card-top');
  const title = element('h3');
  if (rank !== null) title.append(element('span', 'rank', String(rank + 1).padStart(2, '0')));
  title.append(document.createTextNode(context.title));
  top.append(title);
  if (deletable) {
    const remove = element('button', 'delete-button', 'Delete');
    remove.type = 'button';
    remove.setAttribute('aria-label', `Delete chunk: ${context.title}`);
    remove.addEventListener('click', async () => {
      if (!window.confirm(`Delete this stored chunk from “${context.title}”?`)) return;
      remove.disabled = true;
      try {
        await request(`/documents/${context.id}`, { method: 'DELETE' });
        documents = documents.filter((item) => item.id !== context.id);
        renderDocuments();
        await refreshStatus();
        // Do not retain search evidence from a deleted chunk.
        empty($('search-results'), 'Your library changed.', 'Run your search again to refresh the results.');
        $('search-summary').textContent = 'Ready when you are';
        empty($('answer-area'), 'Your library changed.', 'Ask again to retrieve current source passages.');
        notice('Document chunk deleted and saved.');
      } catch (error) { notice(error.message, true); remove.disabled = false; }
    });
    top.append(remove);
  } else {
    top.append(element('span', 'distance', `distance ${Number(context.distance).toFixed(3)}`));
  }
  card.append(top);
  const text = context.text;
  card.append(element('p', 'passage', text.length > 330 ? `${text.slice(0, 330)}…` : text));
  if (text.length > 330) {
    const details = element('details');
    details.append(element('summary', '', 'Read full passage'), element('p', 'passage', text));
    card.append(details);
  }
  return card;
}

async function busy(form, label, work) {
  const button = form.querySelector('button[type="submit"]');
  const original = [...button.childNodes].map((node) => node.cloneNode(true));
  button.disabled = true;
  button.textContent = label;
  form.setAttribute('aria-busy', 'true');
  notice('');
  try { await work(); }
  catch (error) { notice(error.message || 'Unable to reach the server.', true); }
  finally {
    button.disabled = false;
    button.replaceChildren(...original);
    form.removeAttribute('aria-busy');
  }
}

$('search-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const question = $('search-question').value.trim();
  if (!question) { notice('Enter a topic or question.', true); return; }
  const k = Number($('search-k').value);
  busy(event.currentTarget, 'Searching…', async () => {
    const results = $('search-results');
    results.replaceChildren(element('p', 'busy', 'Finding relevant passages…'));
    try {
      const algorithm = $('search-algorithm').value;
      const data = await post('/search', { question, k, algorithm });
      $('search-summary').textContent = `${data.contexts.length} matching passage${data.contexts.length === 1 ? '' : 's'} · ${{ hnsw: 'HNSW', 'kd-tree': 'KD-tree', 'brute-force': 'Brute force' }[data.algorithm]} · lower distance is closer`;
      if (!data.contexts.length) empty(results, 'No relevant passage found.', 'Try rephrasing your question or add a document about this topic.');
      else results.replaceChildren(...data.contexts.map((context, index) => passageCard(context, index)));
    } catch (error) { empty(results, 'Search could not finish.', 'Check the message above and try again.'); throw error; }
  });
});

function renderDocuments() {
  const filter = $('document-filter').value.trim().toLowerCase();
  const selected = documents.filter((item) => `${item.title} ${item.text}`.toLowerCase().includes(filter));
  const pages = Math.max(1, Math.ceil(selected.length / pageSize));
  page = Math.min(page, pages - 1);
  $('library-summary').textContent = `${selected.length.toLocaleString()} chunk${selected.length === 1 ? '' : 's'}${filter ? ' matching your filter' : ' in your library'}`;
  $('page-label').textContent = `Page ${page + 1} of ${pages}`;
  $('previous-page').disabled = page === 0;
  $('next-page').disabled = page >= pages - 1;
  if (!selected.length) empty($('document-list'), filter ? 'No matching documents.' : 'Your library is empty.', filter ? 'Try a different title or keyword.' : 'Add your first document using the form.');
  else $('document-list').replaceChildren(...selected.slice(page * pageSize, (page + 1) * pageSize).map((item) => passageCard(item, null, true)));
}

async function loadDocuments() {
  $('refresh-documents').disabled = true;
  try {
    documents = await request('/documents');
    // Newest inserted IDs first, independent of snapshot dictionary ordering.
    documents.sort((a, b) => b.id - a.id);
    loadedDocuments = true;
    renderDocuments();
  } catch (error) { notice(error.message, true); }
  finally { $('refresh-documents').disabled = false; }
}

$('document-filter').addEventListener('input', () => { page = 0; renderDocuments(); });
$('refresh-documents').addEventListener('click', () => { loadDocuments(); refreshStatus(); });
$('previous-page').addEventListener('click', () => { page -= 1; renderDocuments(); });
$('next-page').addEventListener('click', () => { page += 1; renderDocuments(); });
$('document-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const title = $('document-title').value.trim();
  const text = $('document-text').value.trim();
  if (!title || !text) { notice('Enter both a title and document text.', true); return; }
  busy(event.currentTarget, 'Embedding & saving…', async () => {
    const result = await post('/documents', { title, text });
    $('document-title').value = '';
    $('document-text').value = '';
    $('document-filter').value = '';
    page = 0;
    await Promise.all([refreshStatus(), loadDocuments()]);
    notice(`Saved ${result.ids.length} document chunk${result.ids.length === 1 ? '' : 's'}. Your text is ready to search.`);
  });
});

$('ask-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const question = $('ask-question').value.trim();
  if (!question) { notice('Enter a question.', true); return; }
  const k = Number($('ask-k').value);
  busy(event.currentTarget, 'Thinking…', async () => {
    const area = $('answer-area');
    area.replaceChildren(element('p', 'busy', 'Retrieving evidence and generating an answer. This can take a minute…'));
    try {
      const data = await post('/ask', { question, k });
      const answer = element('article', data.abstained ? 'result-card answer-card abstained' : 'result-card answer-card');
      answer.append(element('p', 'answer-label', data.abstained ? 'NO RELEVANT EVIDENCE' : `ANSWER · GROQ · ${data.model}`), element('p', 'answer-text', data.answer));
      area.replaceChildren(answer);
      if (data.contexts.length) {
        area.append(element('h3', '', `Source passages (${data.contexts.length})`));
        data.contexts.forEach((context, index) => area.append(passageCard(context, index)));
      }
    } catch (error) { empty(area, 'An answer is not available.', 'You can still use semantic search without an answer-generation model.'); throw error; }
  });
});

document.querySelectorAll('[data-view]').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-view]').forEach((item) => {
      const active = item === button;
      item.classList.toggle('active', active);
      if (active) item.setAttribute('aria-current', 'page');
      else item.removeAttribute('aria-current');
    });
    document.querySelectorAll('.view').forEach((view) => { view.hidden = view.id !== `view-${button.dataset.view}`; });
    notice('');
    if (button.dataset.view === 'documents' && !loadedDocuments) loadDocuments();
  });
});

refreshStatus();
