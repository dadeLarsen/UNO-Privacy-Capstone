import { useState } from 'react'
import { styleLabels, type Job, type RewriteStyle } from './api'

const styleIcons: Record<RewriteStyle, string> = {
  paraphrase: '↗',
  simplify_for_kids: '☼',
  simplify_for_adults: '◌',
}

// What the button that starts a rewrite says for each style, before and after what it rewrites:
// "Simplify the selection for kids".
const styleActions: Record<RewriteStyle, [string, string]> = {
  paraphrase: ['Paraphrase', ''],
  simplify_for_kids: ['Simplify', ' for kids'],
  simplify_for_adults: ['Simplify', ' for adults'],
}

type Props = {
  busy: boolean
  // What a rewrite takes: the selection or the whole policy.
  scope: string
  // Set while sentences are selected: empties the selection, so the scope is the whole policy again.
  onClearSelection?: () => void
  // The job that runs now, of either kind.
  activeJob: Job | null
  onRewrite: (style: RewriteStyle, label: string) => void
  onSummarize: () => void
  onCancel: () => void
}

// The toolbar that starts a rewrite or the summaries and shows the running job. The rewrite itself
// is in the sentence map, beside the sentences it came from.
function RewritePanel({ busy, scope, onClearSelection, activeJob, onRewrite, onSummarize, onCancel }: Props) {
  const [style, setStyle] = useState<RewriteStyle>('simplify_for_adults')
  const [verb, audience] = styleActions[style]
  const progress = activeJob?.progress

  return (
    <section aria-label="Rewrite" className="rewrite-panel">
      <div aria-label="Reading style" className="rewrite-options" role="group">
        {(Object.keys(styleLabels) as RewriteStyle[]).map((option) => (
          <button
            aria-pressed={style === option}
            className={style === option ? 'selected' : ''}
            onClick={() => setStyle(option)}
            key={option}
            type="button"
          >
            <span aria-hidden="true">{styleIcons[option]}</span>{styleLabels[option]}
          </button>
        ))}
      </div>
      <p className="rewrite-scope">
        Scope: {scope}
        {onClearSelection && (
          <button className="clear-selection" onClick={onClearSelection} type="button">Clear selection</button>
        )}
      </p>
      <div className="rewrite-actions">
        <button
          className="primary-button"
          onClick={() => onRewrite(style, styleLabels[style])}
          disabled={busy}
          type="button"
        >
          <span>{verb} {onClearSelection ? 'the selection' : 'the policy'}{audience}</span><b>→</b>
        </button>
        <button className="secondary-button" onClick={onSummarize} disabled={busy} type="button">
          Summarize categories
        </button>
      </div>
      {activeJob && (
        <div className="rewrite-progress">
          <span role="status">{progress?.message ?? 'Working…'}</span>
          {progress && progress.total > 1 && <progress max={progress.total} value={progress.completed} />}
          <button className="secondary-button" onClick={onCancel} type="button">
            Cancel {activeJob.kind}
          </button>
        </div>
      )}
    </section>
  )
}

export default RewritePanel
