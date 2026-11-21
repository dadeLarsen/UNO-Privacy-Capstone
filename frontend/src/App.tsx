import { useEffect, useState, type ChangeEvent, type CSSProperties } from 'react'
import {
  ApiError,
  cancelJob,
  classify,
  createPolicy,
  getCategories,
  getJob,
  getResults,
  startRewrite,
  startSummaries,
  uploadPolicy,
  type AnalysisSnapshot,
  type Category,
  type CategoryList,
  type Job,
  type RewriteStyle,
  type Sentence,
} from './api'
import './App.css'
import RewritePanel from './RewritePanel'

const examplePolicy = `At BrightCart, your privacy is important to us. When you sign up, place an order, or contact our support team, we collect personal information such as your name, email address, shipping details, and payment method to process your requests and improve your shopping experience.

We may share certain information, such as your email address, with trusted third-party marketing partners who assist us in delivering personalized promotions. These partners are required to maintain confidentiality and are not permitted to use your data for other purposes.

You have the ability to manage your communication preferences at any time. To stop receiving promotional emails, simply click the unsubscribe link in any message or update your account settings.`

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function previewSentences(text: string): Sentence[] {
  return [...text.matchAll(/[^.!?]+[.!?]*/g)]
    .map((match, id) => {
      const value = match[0].trim()
      return { id, text: value, start: 0, end: value.length, label_id: null, label: null }
    })
    .filter((sentence) => sentence.text.length > 0)
}

function tintOf(category: Category): string {
  return category.tint ?? `${category.color}1a`
}

function App() {
  const [policy, setPolicy] = useState(examplePolicy)
  const [policyId, setPolicyId] = useState<string | null>(null)
  const [categories, setCategories] = useState<Category[]>([])
  const [snapshot, setSnapshot] = useState<AnalysisSnapshot | null>(null)
  const [activeCategories, setActiveCategories] = useState<Set<number>>(() => new Set())
  const [selectedSentences, setSelectedSentences] = useState<Set<number>>(() => new Set())
  const [highlighted, setHighlighted] = useState(true)
  const [activeJob, setActiveJob] = useState<Job | null>(null)
  // The latest rewrite job as last polled. It stays after the job ends, to say whether the rewrite is whole.
  const [rewriteJob, setRewriteJob] = useState<Job | null>(null)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('Ready to analyze')
  const [error, setError] = useState<string | null>(null)

  function showCategories(body: CategoryList) {
    setCategories(body.categories)
    setActiveCategories(new Set(body.categories.map((category) => category.id)))
  }

  useEffect(() => {
    getCategories()
      .then(showCategories)
      .catch((cause: unknown) => setError(`API unavailable: ${messageOf(cause)}`))
  }, [])

  function forgetPolicy() {
    setPolicyId(null)
    setSnapshot(null)
    setSelectedSentences(new Set())
    setRewriteJob(null)
  }

  function editPolicy(value: string) {
    setPolicy(value)
    forgetPolicy()
    setError(null)
    setStatus('Ready to analyze')
  }

  function showFailure(cause: unknown, failedStatus: string) {
    if (cause instanceof ApiError && cause.code === 'not_found') {
      // The API dropped the policy or its job. The text is still on the page, so the next Analyze sends it again.
      forgetPolicy()
      setError(
        'The API no longer has this policy in memory. It clears a policy after an hour without use, ' +
          'and when it restarts. Click Analyze categories to run it again.',
      )
      setStatus('Ready to analyze')
      return
    }
    setError(messageOf(cause))
    setStatus(failedStatus)
  }

  function toggleCategory(id: number) {
    setActiveCategories((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  function toggleSentence(id: number) {
    if (!snapshot?.classification) return
    setSelectedSentences((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  async function ensurePolicy(): Promise<string> {
    if (policyId) return policyId
    const created = await createPolicy(policy)
    setPolicyId(created.id)
    return created.id
  }

  async function classifyAndShow(id: string) {
    // The list is empty when the page opened before the model service was ready.
    if (!categories.length) showCategories(await getCategories())
    const result = await classify(id)
    setSnapshot(await getResults(id))
    return result
  }

  async function analyze() {
    setBusy(true)
    setError(null)
    setStatus('Analyzing…')
    try {
      const result = await classifyAndShow(await ensurePolicy())
      setStatus(`Analyzed locally · ${result.sentences.length} sentences`)
      if (activeCategories.size === 0 && categories.length) {
        setActiveCategories(new Set(categories.map((category) => category.id)))
      }
    } catch (cause: unknown) {
      showFailure(cause, 'Analysis failed')
    } finally {
      setBusy(false)
    }
  }

  async function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return
    setBusy(true)
    setError(null)
    setStatus(`Reading ${file.name}…`)
    try {
      const text = await file.text()
      setPolicy(text)
      forgetPolicy()
      const uploaded = await uploadPolicy(file)
      setPolicyId(uploaded.id)
      const result = await classifyAndShow(uploaded.id)
      setStatus(`Analyzed ${file.name} · ${result.sentences.length} sentences`)
    } catch (cause: unknown) {
      setError(messageOf(cause))
      setStatus('Upload failed')
    } finally {
      setBusy(false)
      event.target.value = ''
    }
  }

  async function pollJob(jobId: string, id: string) {
    for (;;) {
      const job = await getJob(jobId)
      if (job.kind === 'rewrite') setRewriteJob(job)
      if (job.status === 'queued' || job.status === 'running') {
        setActiveJob(job)
        await new Promise((resolve) => setTimeout(resolve, 250))
        continue
      }
      setActiveJob(null)
      // A cancelled or failed rewrite keeps the sections it finished, so every ending loads the results.
      setSnapshot(await getResults(id))
      const name = job.kind === 'summary' ? 'Summary' : 'Rewrite'
      const sections = job.kind === 'rewrite' && job.progress ? job.progress : null
      if (job.status === 'succeeded') {
        setStatus(job.kind === 'summary' ? 'Summaries ready' : 'Rewrite ready')
      } else if (job.status === 'failed') {
        setError(job.error?.message ?? `${name} failed`)
        setStatus(sections ? `${name} failed at section ${sections.completed + 1} of ${sections.total}` : `${name} failed`)
      } else {
        setStatus(
          sections ? `${name} cancelled after ${sections.completed} of ${sections.total} sections` : `${name} cancelled`,
        )
      }
      return
    }
  }

  async function summarize() {
    setBusy(true)
    setError(null)
    try {
      const id = await ensurePolicy()
      if (!snapshot?.classification) await classifyAndShow(id)
      const job = await startSummaries(id)
      setActiveJob(job)
      setStatus('Summarizing by category…')
      await pollJob(job.id, id)
    } catch (cause: unknown) {
      showFailure(cause, 'Summary failed')
      setActiveJob(null)
    } finally {
      setBusy(false)
    }
  }

  async function rewrite(style: RewriteStyle, label: string) {
    setBusy(true)
    setError(null)
    try {
      const id = await ensurePolicy()
      if (!snapshot?.classification) await classifyAndShow(id)
      const selected = selectedSentences.size
        ? [...selectedSentences].sort((left, right) => left - right)
        : undefined
      const job = await startRewrite(id, style, selected)
      setActiveJob(job)
      setStatus(`Generating ${label}…`)
      await pollJob(job.id, id)
    } catch (cause: unknown) {
      showFailure(cause, 'Rewrite failed')
      setActiveJob(null)
    } finally {
      setBusy(false)
    }
  }

  async function cancelActiveJob() {
    if (!activeJob) return
    try {
      // The poll that follows sees the cancelled job, loads what it finished, and says how far it was.
      await cancelJob(activeJob.id)
      setActiveJob(null)
    } catch (cause: unknown) {
      setError(messageOf(cause))
    }
  }

  const categoryMap = new Map(categories.map((category) => [category.id, category]))
  const counts = new Map(snapshot?.classification?.counts.map((count) => [count.label_id, count.count]) ?? [])
  const sentences = snapshot?.classification?.sentences ?? previewSentences(policy)
  const analyzed = snapshot?.classification !== null && snapshot?.classification !== undefined

  return (
    <div className="app-shell">
      <aside className="category-rail">
        <div className="rail-label">Categories</div>
        <div className="category-list">
          {categories.map((category) => (
            <label className="category-chip" key={category.id} style={{ backgroundColor: category.color }}>
              <input
                checked={activeCategories.has(category.id)}
                onChange={() => toggleCategory(category.id)}
                type="checkbox"
              />
              <span>{category.display_name}</span>
              <strong>{counts.get(category.id) ?? 0}</strong>
            </label>
          ))}
        </div>
        <div className="rail-footnote">Local analysis<br /><span>Nothing leaves this device</span></div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Privacy Policy Reader</p>
            <h1>See what a policy really says.</h1>
            <p className="api-status">{status}</p>
          </div>
          <span className="offline-badge"><i /> Offline mode</span>
        </header>

        {error && <div className="api-error" role="alert">{error}</div>}

        <section className="intro-panel">
          <div className="panel-heading">
            <div><span className="step">01</span><h2>Bring in a policy</h2></div>
            <span className="status-dot">{analyzed ? 'Analyzed locally' : 'Ready to analyze'}</span>
          </div>
          <textarea
            aria-label="Privacy policy text"
            value={policy}
            onChange={(event) => editPolicy(event.target.value)}
          />
          <div className="action-row">
            <span className="character-count">
              {policy.length.toLocaleString()} characters · {sentences.length} sentences
            </span>
            <div className="button-row">
              <label className="secondary-button">
                Open .txt
                <input type="file" accept=".txt,text/plain" onChange={onFileChange} disabled={busy} />
              </label>
              <button className="primary-button" onClick={analyze} disabled={busy} type="button">
                <span>Analyze categories</span><b>→</b>
              </button>
            </div>
          </div>
        </section>

        <div className="content-grid">
          <section className="results-panel">
            <div className="panel-heading result-heading">
              <div><span className="step">02</span><h2>Sentence map</h2></div>
              <label className="toggle-label">
                <input
                  checked={highlighted && analyzed}
                  disabled={!analyzed}
                  onChange={(event) => setHighlighted(event.target.checked)}
                  type="checkbox"
                />
                <span className="toggle" /> Highlight categories
              </label>
            </div>
            {analyzed && (
              <p className="selection-hint">
                {selectedSentences.size
                  ? `${selectedSentences.size} sentence${selectedSentences.size === 1 ? '' : 's'} selected for rewriting. Click again to deselect.`
                  : 'Click sentences to select them for rewriting. With none selected, the whole policy is used.'}
              </p>
            )}
            <div className="sentence-list">
              {sentences.map((sentence) => {
                const category = sentence.label_id === null ? undefined : categoryMap.get(sentence.label_id)
                const isVisible = highlighted && category !== undefined && activeCategories.has(sentence.label_id!)
                const isSelected = selectedSentences.has(sentence.id)
                return (
                  <div
                    aria-pressed={analyzed ? isSelected : undefined}
                    className={`sentence ${isVisible ? 'is-highlighted' : ''} ${isSelected ? 'is-selected' : ''}`}
                    key={sentence.id}
                    onClick={() => toggleSentence(sentence.id)}
                    onKeyDown={(event) => {
                      if (analyzed && (event.key === 'Enter' || event.key === ' ')) {
                        event.preventDefault()
                        toggleSentence(sentence.id)
                      }
                    }}
                    role={analyzed ? 'button' : undefined}
                    style={
                      category
                        ? { '--sentence-tint': tintOf(category), '--sentence-color': category.color } as CSSProperties
                        : undefined
                    }
                    tabIndex={analyzed ? 0 : undefined}
                  >
                    <span className="sentence-number">{String(sentence.id + 1).padStart(2, '0')}</span>
                    <p>{sentence.text}</p>
                    {category && <span className="sentence-tag">{category.display_name}</span>}
                    {sentence.error && <span className="sentence-error">{sentence.error}</span>}
                  </div>
                )
              })}
              {!sentences.length && <div className="empty-state">Paste policy text above to see sentence-level results.</div>}
            </div>

            {snapshot?.summaries.length ? (
              <div className="summary-list">
                <h3>Category summaries</h3>
                {snapshot.summaries.map((summary) => (
                  <div className="summary-item" key={summary.label_id}>
                    <strong>{summary.category}</strong>
                    <p>{summary.text}</p>
                    <span className="source-link">
                      Sources: {summary.source_sentence_ids.map((id) => id + 1).join(', ')}
                    </span>
                  </div>
                ))}
              </div>
            ) : null}
          </section>

          <RewritePanel
            busy={busy}
            scope={
              selectedSentences.size
                ? `${selectedSentences.size} selected ${selectedSentences.size === 1 ? 'sentence' : 'sentences'}`
                : 'the whole policy'
            }
            activeJob={activeJob}
            rewriteJob={rewriteJob}
            rewrites={snapshot?.rewrites ?? []}
            onRewrite={rewrite}
            onSummarize={summarize}
            onCancel={cancelActiveJob}
          />
        </div>
      </main>
    </div>
  )
}

export default App
