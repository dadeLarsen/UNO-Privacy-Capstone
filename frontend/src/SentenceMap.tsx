import { Fragment, useEffect, useState, type CSSProperties } from 'react'
import { styleLabels, type Category, type Job, type Rewrite, type Sentence, type Summary } from './api'

type View = 'original' | 'both' | 'rewrite'

const viewLabels: Record<View, string> = { original: 'Original', both: 'Both', rewrite: 'Rewrite' }

// One section of the rewrite on the page: its sentences, and its rewrite once it has one.
type Block = { ids: number[]; rewrite?: Rewrite }

// A stretch of the policy. A row with a block runs from the block's first sentence to its last.
type Row = { sentences: Sentence[]; block?: Block; index: number }

type Props = {
  // Null until the API has labeled the policy.
  sentences: Sentence[] | null
  // What the map says in place of the sentences until then.
  waiting: string
  // Set when the labeling failed: sends the same policy again.
  onRetry?: () => void
  categories: Map<number, Category>
  hiddenCategories: Set<number>
  highlighted: boolean
  onHighlight: (highlighted: boolean) => void
  selected: Set<number>
  // A click or a key on a sentence; `range` is true when Shift was held.
  onPick: (id: number, range: boolean) => void
  // The latest rewrite job as last polled, also after it ended.
  rewriteJob: Job | null
  // The rewrite in the policy's results.
  rewrites: Rewrite[]
  summaries: Summary[]
}

function tintOf(category: Category): string {
  return category.tint ?? `${category.color}1a`
}

// The sections of the rewrite on the page, in policy order. A job names all of its sections from its
// start, so the ones it has not reached, or never reached, are here without a rewrite.
function blocksOf(job: Job | null, kept: Rewrite[]): Block[] {
  const own = job?.result?.kind === 'rewrite' ? job.result.rewrites : null
  const running = job?.status === 'queued' || job?.status === 'running'
  // A job that ended before its first section leaves the policy's earlier rewrite in place.
  if (!job?.sections || !(own || running)) {
    return kept.map((rewrite) => ({ ids: rewrite.source_sentence_ids, rewrite }))
  }
  return job.sections.map((ids, index) => ({ ids, rewrite: own?.[index] }))
}

// Cut the policy into rows, so that each block stands level with its sentences. The sentences between
// two blocks get a row of their own. A selection can skip sentences, and a block's row holds those too.
function rowsOf(sentences: Sentence[], blocks: Block[]): Row[] {
  const rows: Row[] = []
  let next = 0
  blocks.forEach((block, index) => {
    const first = sentences.findIndex((sentence) => sentence.id === block.ids[0])
    const last = sentences.findIndex((sentence) => sentence.id === block.ids[block.ids.length - 1])
    if (first > next) rows.push({ sentences: sentences.slice(next, first), index: -1 })
    rows.push({ sentences: sentences.slice(first, last + 1), block, index })
    next = last + 1
  })
  if (next < sentences.length) rows.push({ sentences: sentences.slice(next), index: -1 })
  return rows
}

// Why a section has no rewrite.
function pendingNote(job: Job | null, index: number): string {
  if (job?.status === 'cancelled') return 'Not rewritten: the rewrite was cancelled.'
  if (job?.status === 'failed') return 'Not rewritten: the rewrite failed.'
  return index === (job?.progress?.completed ?? 0) ? 'Rewriting this section…' : 'Waiting for its turn.'
}

// Sentence IDs in runs that follow one another: 4, 5 and 9 give [[4, 5], [9]].
function runsOf(ids: number[]): number[][] {
  const runs: number[][] = []
  for (const id of ids) {
    const run = runs[runs.length - 1]
    if (run && id === run[run.length - 1] + 1) run.push(id)
    else runs.push([id])
  }
  return runs
}

// One link for each run of source sentences. A link brings its run into view and outlines every source.
function SourceLinks({ ids, onCite }: { ids: number[]; onCite: (ids: number[], first: number) => void }) {
  return (
    <span className="source-link">
      {ids.length === 1 ? 'Source: ' : 'Sources: '}
      {runsOf(ids).map((run, index) => (
        <Fragment key={run[0]}>
          {index > 0 && ', '}
          <a
            href={`#sentence-${run[0] + 1}`}
            onClick={(event) => {
              event.preventDefault()
              onCite(ids, run[0])
            }}
          >
            {run.length === 1 ? `sentence ${run[0] + 1}` : `sentences ${run[0] + 1}–${run[run.length - 1] + 1}`}
          </a>
        </Fragment>
      ))}
    </span>
  )
}

function SentenceMap({
  sentences,
  waiting,
  onRetry,
  categories,
  hiddenCategories,
  highlighted,
  onHighlight,
  selected,
  onPick,
  rewriteJob,
  rewrites,
  summaries,
}: Props) {
  const [view, setView] = useState<View>('both')
  // The sources of the link followed last, which the map outlines, and the sentence to bring into view.
  const [cited, setCited] = useState<{ ids: number[]; first: number } | null>(null)

  useEffect(() => {
    if (!cited) return
    const target = document.getElementById(`sentence-${cited.first + 1}`)
    target?.scrollIntoView({ block: 'center' })
    target?.focus({ preventScroll: true })
  }, [cited])

  if (!sentences) {
    return (
      <section className="policy-rows is-single">
        <div className="policy-card empty-state">
          <p>{waiting}</p>
          {onRetry && <button className="secondary-button" onClick={onRetry} type="button">Try again</button>}
        </div>
      </section>
    )
  }

  const blocks = blocksOf(rewriteJob, rewrites)
  // Without a rewrite there is nothing to switch between.
  const shown = blocks.length ? view : 'original'
  const written = blocks.flatMap((block) => (block.rewrite ? [block.rewrite] : []))
  // A warning on every section, such as the note on the style prompt, shows once above them.
  const shared = written.length
    ? written[0].warnings.filter((warning) => written.every((item) => item.warnings.includes(warning)))
    : []
  const style = rewriteJob?.result?.kind === 'rewrite' ? rewriteJob.result.style : null
  const ending = rewriteJob?.status === 'cancelled' ? ', cancelled' : rewriteJob?.status === 'failed' ? ', failed' : ' so far'

  function cite(ids: number[], first: number) {
    // The Rewrite view hides the original, where the link leads.
    if (view === 'rewrite') setView('both')
    setCited({ ids, first })
  }

  function sentenceRow(sentence: Sentence) {
    const category = sentence.label_id === null ? undefined : categories.get(sentence.label_id)
    const isVisible = highlighted && category !== undefined && !hiddenCategories.has(category.id)
    const isSelected = selected.has(sentence.id)
    const isCited = cited?.ids.includes(sentence.id) ?? false
    return (
      <div
        aria-pressed={isSelected}
        className={`sentence ${isVisible ? 'is-highlighted' : ''} ${isSelected ? 'is-selected' : ''} ${isCited ? 'is-cited' : ''}`}
        id={`sentence-${sentence.id + 1}`}
        key={sentence.id}
        onClick={(event) => onPick(sentence.id, event.shiftKey)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            onPick(sentence.id, event.shiftKey)
          }
        }}
        onMouseDown={(event) => {
          // A Shift-click would also stretch the browser's text selection across the sentences.
          if (event.shiftKey) event.preventDefault()
        }}
        role="button"
        style={
          category
            ? { '--sentence-tint': tintOf(category), '--sentence-color': category.color } as CSSProperties
            : undefined
        }
        tabIndex={0}
      >
        <span className="sentence-number">{String(sentence.id + 1).padStart(2, '0')}</span>
        <p>{sentence.text}</p>
        {category && <span className="sentence-tag">{category.display_name}</span>}
        {sentence.error && <span className="sentence-error">{sentence.error}</span>}
      </div>
    )
  }

  function blockCell(block: Block, index: number) {
    const own = block.rewrite?.warnings.filter((warning) => !shared.includes(warning)) ?? []
    return (
      <div className={`row-rewrite ${block.rewrite ? '' : 'is-pending'}`}>
        <p>{block.rewrite ? block.rewrite.text : pendingNote(rewriteJob, index)}</p>
        <SourceLinks ids={block.ids} onCite={cite} />
        {own.length > 0 && (
          <ul className="rewrite-warnings">
            {own.map((warning) => <li key={warning}>{warning}</li>)}
          </ul>
        )}
      </div>
    )
  }

  return (
    <>
      {summaries.length > 0 && (
        <details className="summary-list" open>
          <summary>Category summaries</summary>
          {summaries.map((summary) => (
            <div className="summary-item" key={summary.label_id}>
              <strong>{summary.category}</strong>
              <p>{summary.text}</p>
              <SourceLinks ids={summary.source_sentence_ids} onCite={cite} />
            </div>
          ))}
        </details>
      )}

      {blocks.length > 0 && (
        <div className="view-switch">
          <span id="view-switch-label">Show</span>
          <div aria-labelledby="view-switch-label" role="group">
            {(Object.keys(viewLabels) as View[]).map((option) => (
              <button aria-pressed={view === option} key={option} onClick={() => setView(option)} type="button">
                {viewLabels[option]}
              </button>
            ))}
          </div>
        </div>
      )}

      <section className={`policy-rows ${shown === 'both' ? '' : 'is-single'}`}>
        <div className="policy-row policy-heads">
          {shown !== 'rewrite' && (
            <div className="column-head">
              <div className="column-title">
                <h2>{blocks.length ? 'Original' : 'Sentence map'}</h2>
                <label className="toggle-label">
                  <input
                    checked={highlighted}
                    onChange={(event) => onHighlight(event.target.checked)}
                    type="checkbox"
                  />
                  <span className="toggle" /> Highlight categories
                </label>
              </div>
              <p className="selection-hint">
                {selected.size
                  ? `${selected.size} ${selected.size === 1 ? 'sentence' : 'sentences'} selected for rewriting. `
                  : 'With no sentence selected, a rewrite takes the whole policy. '}
                Click a sentence to select or deselect it, and Shift-click another to select every sentence between
                them. On the keyboard, Enter or Space is the click.
              </p>
            </div>
          )}
          {shown !== 'original' && (
            <div className="column-head">
              <div className="column-title">
                <h2>Rewrite{style && ` · ${styleLabels[style]}`}</h2>
                {written.length < blocks.length && (
                  <span className="rewrite-part">{written.length} of {blocks.length} sections{ending}</span>
                )}
              </div>
              {shared.length > 0 && (
                <ul className="rewrite-warnings">
                  {shared.map((warning) => <li key={warning}>{warning}</li>)}
                </ul>
              )}
            </div>
          )}
        </div>
        {rowsOf(sentences, blocks).map((row) =>
          shown === 'rewrite' && !row.block ? null : (
            <div className="policy-row policy-card" key={row.sentences[0].id}>
              {shown !== 'rewrite' && <div className="sentence-list">{row.sentences.map(sentenceRow)}</div>}
              {shown !== 'original' && row.block && blockCell(row.block, row.index)}
            </div>
          ),
        )}
      </section>
    </>
  )
}

export default SentenceMap
