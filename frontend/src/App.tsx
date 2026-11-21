import { useEffect, useRef, useState, type ChangeEvent } from 'react'
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
  type Job,
  type RewriteStyle,
} from './api'
import './App.css'
import RewritePanel from './RewritePanel'
import SentenceMap from './SentenceMap'

const examplePolicy = `At BrightCart, your privacy is important to us. When you sign up, place an order, or contact our support team, we collect personal information such as your name, email address, shipping details, and payment method to process your requests and improve your shopping experience.

We may share certain information, such as your email address, with trusted third-party marketing partners who assist us in delivering personalized promotions. These partners are required to maintain confidentiality and are not permitted to use your data for other purposes.

You have the ability to manage your communication preferences at any time. To stop receiving promotional emails, simply click the unsubscribe link in any message or update your account settings.`

// How long a typed or pasted text must stay unchanged before the page labels it.
const SETTLE_MS = 600
// How long the page waits before it asks again while a local service is still starting.
const RETRY_MS = 2000
const CATEGORIZING = 'Categorizing sentences…'

// The policy on the page. `file` is the file it was opened from, and `typed` marks an edit in the text box.
type Source = { text: string; file?: File; typed?: boolean }

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

// The local service that a failed request has to wait for, or null when the failure has another cause.
function startingService(cause: unknown): string | null {
  // The model service opens its port once its classifier has loaded. Until then the API answers with this code.
  if (cause instanceof ApiError) return cause.code === 'model_service_unavailable' ? 'the model service' : null
  // fetch rejects with a TypeError when nothing answers at the API's address.
  return cause instanceof TypeError ? 'the API' : null
}

function App() {
  const [source, setSource] = useState<Source>({ text: examplePolicy })
  const [policyId, setPolicyId] = useState<string | null>(null)
  const [categories, setCategories] = useState<Category[]>([])
  const [snapshot, setSnapshot] = useState<AnalysisSnapshot | null>(null)
  const [hiddenCategories, setHiddenCategories] = useState<Set<number>>(() => new Set())
  const [selectedSentences, setSelectedSentences] = useState<Set<number>>(() => new Set())
  const [highlighted, setHighlighted] = useState(true)
  const [activeJob, setActiveJob] = useState<Job | null>(null)
  // The latest rewrite job as last polled. It stays after the job ends, to say whether the rewrite is whole.
  const [rewriteJob, setRewriteJob] = useState<Job | null>(null)
  const [busy, setBusy] = useState(false)
  const [analysisFailed, setAnalysisFailed] = useState(false)
  const [status, setStatus] = useState(CATEGORIZING)
  const [error, setError] = useState<string | null>(null)
  // Counts the policies the page has held. A job compares it with the count at its start, and stops
  // writing to the page once the page holds another policy.
  const generation = useRef(0)
  // The sentence last clicked or keyed, where a Shift range starts.
  const anchor = useRef<number | null>(null)
  const sentences = snapshot?.classification?.sentences ?? null

  // Label the policy as soon as it comes in. The cleanup drops the answer for a text that changed meanwhile.
  useEffect(() => {
    if (!source.file && !source.text.trim()) return
    let current = true
    const timer = setTimeout(async () => {
      let id: string | undefined
      for (;;) {
        try {
          const list = await getCategories()
          if (!current) return
          setCategories(list.categories)
          setStatus(CATEGORIZING)
          id ??= (source.file ? await uploadPolicy(source.file) : await createPolicy(source.text)).id
          const result = await classify(id)
          const results = await getResults(id)
          if (!current) return
          setPolicyId(id)
          setSnapshot(results)
          const count = result.sentences.length
          setStatus(`Analyzed ${source.file?.name ?? 'locally'} · ${count} ${count === 1 ? 'sentence' : 'sentences'}`)
          return
        } catch (cause: unknown) {
          if (!current) return
          const service = startingService(cause)
          if (!service) {
            setError(messageOf(cause))
            setStatus('Analysis failed')
            setAnalysisFailed(true)
            return
          }
          setStatus(`Waiting for ${service} to start…`)
          await new Promise((resolve) => setTimeout(resolve, RETRY_MS))
          if (!current) return
        }
      }
    }, source.typed ? SETTLE_MS : 0)
    return () => {
      current = false
      clearTimeout(timer)
    }
  }, [source])

  // Put another policy on the page, or the same one again. Everything that belonged to the last one goes.
  function load(next: Source) {
    generation.current += 1
    anchor.current = null
    setSource(next)
    setPolicyId(null)
    setSnapshot(null)
    setSelectedSentences(new Set())
    setActiveJob(null)
    setRewriteJob(null)
    setBusy(false)
    setAnalysisFailed(false)
    setError(null)
    setStatus(next.file || next.text.trim() ? CATEGORIZING : 'Paste a policy to begin')
  }

  function showFailure(cause: unknown, failedStatus: string) {
    if (cause instanceof ApiError && cause.code === 'not_found') {
      // The API dropped the policy or its job. The text is still on the page, so it goes to the API again.
      load({ text: source.text, file: source.file })
      setError(
        'The API no longer had this policy in memory. It clears a policy after an hour without use, ' +
          'and when it restarts. The page has sent the text again; repeat the request when the labels are back.',
      )
      return
    }
    setError(messageOf(cause))
    setStatus(failedStatus)
  }

  function toggleCategory(id: number) {
    setHiddenCategories((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  // A click or a key on a sentence. With Shift it selects every sentence from the last one picked to this one.
  function pickSentence(id: number, range: boolean) {
    const from = range ? anchor.current : null
    anchor.current = id
    setSelectedSentences((current) => {
      const next = new Set(current)
      if (from !== null) {
        for (const sentence of sentences ?? []) {
          if (sentence.id >= Math.min(from, id) && sentence.id <= Math.max(from, id)) next.add(sentence.id)
        }
      } else if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  // A category's Select button: adds its sentences to the selection, or takes them out when all are in it.
  function toggleCategorySelection(ids: number[]) {
    setSelectedSentences((current) => {
      const next = new Set(current)
      const allSelected = ids.every((id) => next.has(id))
      for (const id of ids) {
        if (allSelected) next.delete(id)
        else next.add(id)
      }
      return next
    })
  }

  async function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const input = event.target
    const file = input.files?.[0]
    if (!file) return
    try {
      // The text box shows the file's text. The API reads the file itself, and refuses one that is not UTF-8 text.
      load({ text: await file.text(), file })
    } catch (cause: unknown) {
      setError(messageOf(cause))
      setStatus('Upload failed')
    } finally {
      input.value = ''
    }
  }

  async function pollJob(jobId: string, id: string, mine: number) {
    for (;;) {
      const job = await getJob(jobId)
      if (mine !== generation.current) {
        // The page holds another policy now, so this job would only keep the model service busy.
        // The cancel fails when the job has ended already, which is as good.
        cancelJob(jobId).catch(() => undefined)
        return
      }
      if (job.kind === 'rewrite') setRewriteJob(job)
      if (job.status === 'queued' || job.status === 'running') {
        setActiveJob(job)
        await new Promise((resolve) => setTimeout(resolve, 250))
        continue
      }
      // A cancelled or failed rewrite keeps the sections it finished, so every ending loads the results.
      const results = await getResults(id)
      if (mine !== generation.current) return
      setActiveJob(null)
      setSnapshot(results)
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

  // Start a summary or a rewrite job for the labeled policy, and follow it to its end.
  async function runJob(start: (id: string) => Promise<Job>, startedStatus: string, failedStatus: string) {
    if (!policyId) return
    const mine = generation.current
    setBusy(true)
    setError(null)
    try {
      const job = await start(policyId)
      if (mine === generation.current) {
        setActiveJob(job)
        setStatus(startedStatus)
      }
      await pollJob(job.id, policyId, mine)
    } catch (cause: unknown) {
      if (mine === generation.current) {
        setActiveJob(null)
        showFailure(cause, failedStatus)
      }
    } finally {
      if (mine === generation.current) setBusy(false)
    }
  }

  function summarize() {
    void runJob((id) => startSummaries(id), 'Summarizing by category…', 'Summary failed')
  }

  function rewrite(style: RewriteStyle, label: string) {
    const selected = selectedSentences.size
      ? [...selectedSentences].sort((left, right) => left - right)
      : undefined
    void runJob((id) => startRewrite(id, style, selected), `Rewriting · ${label}…`, 'Rewrite failed')
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
  const hasPolicy = Boolean(source.file) || source.text.trim().length > 0

  return (
    <div className="app-shell">
      <aside className="category-rail">
        <div className="rail-label">Categories</div>
        <p className="rail-hint">
          A checkbox shows or hides a category's highlight. Select adds its sentences to the selection, and
          Deselect takes them out again.
        </p>
        <div className="category-list">
          {categories.map((category) => {
            const count = counts.get(category.id) ?? 0
            const ids = (sentences ?? []).filter((sentence) => sentence.label_id === category.id).map((sentence) => sentence.id)
            const allSelected = ids.length > 0 && ids.every((id) => selectedSentences.has(id))
            return (
              <div className="category-row" key={category.id}>
                <label className="category-chip" style={{ backgroundColor: category.color }}>
                  <input
                    checked={!hiddenCategories.has(category.id)}
                    onChange={() => toggleCategory(category.id)}
                    type="checkbox"
                  />
                  <span>{category.display_name}</span>
                  <strong>{count}</strong>
                </label>
                <button
                  aria-label={`${allSelected ? 'Deselect' : 'Select'} the ${count} ${category.display_name} ${count === 1 ? 'sentence' : 'sentences'}`}
                  className={`category-select ${allSelected ? 'is-on' : ''}`}
                  disabled={ids.length === 0}
                  onClick={() => toggleCategorySelection(ids)}
                  type="button"
                >
                  {allSelected ? 'Deselect' : 'Select'}
                </button>
              </div>
            )
          })}
        </div>
        <div className="rail-footnote">Local analysis<br /><span>Nothing leaves this device</span></div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Privacy Policy Reader</p>
            <h1>See what a policy really says.</h1>
            <p className="api-status" role="status">{status}</p>
          </div>
          <span className="offline-badge"><i /> Offline mode</span>
        </header>

        {error && <div className="api-error" role="alert">{error}</div>}

        <section className="intro-panel">
          <h2>Bring in a policy</h2>
          <textarea
            aria-label="Privacy policy text"
            value={source.text}
            onChange={(event) => load({ text: event.target.value, typed: true })}
          />
          <div className="action-row">
            <span className="character-count">
              {source.text.length.toLocaleString()} characters
              {sentences && ` · ${sentences.length} ${sentences.length === 1 ? 'sentence' : 'sentences'}`}
            </span>
            <label className="secondary-button">
              Open .txt
              <input type="file" accept=".txt,text/plain" onChange={onFileChange} />
            </label>
          </div>
        </section>

        <RewritePanel
          busy={busy || !sentences}
          scope={
            selectedSentences.size
              ? `${selectedSentences.size} selected ${selectedSentences.size === 1 ? 'sentence' : 'sentences'}`
              : 'the whole policy'
          }
          onClearSelection={selectedSentences.size ? () => setSelectedSentences(new Set()) : undefined}
          activeJob={activeJob}
          onRewrite={rewrite}
          onSummarize={summarize}
          onCancel={cancelActiveJob}
        />

        {/* The key gives each policy a map of its own, so no view or source outline carries over. */}
        <SentenceMap
          key={policyId ?? 'unlabeled'}
          sentences={sentences}
          waiting={hasPolicy ? status : 'Paste policy text above to see sentence-level results.'}
          onRetry={analysisFailed ? () => load({ text: source.text, file: source.file }) : undefined}
          categories={categoryMap}
          hiddenCategories={hiddenCategories}
          highlighted={highlighted}
          onHighlight={setHighlighted}
          selected={selectedSentences}
          onPick={pickSentence}
          rewriteJob={rewriteJob}
          rewrites={snapshot?.rewrites ?? []}
          summaries={snapshot?.summaries ?? []}
        />
      </main>
    </div>
  )
}

export default App
