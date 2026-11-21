import { useState, type CSSProperties } from 'react'
import type { Category, Sentence } from './api'

type Props = {
  // Null until the API has labeled the policy.
  sentences: Sentence[] | null
  // What the map says in place of the sentences until then.
  waiting: string
  // Set when the labeling failed: sends the same policy again.
  onRetry?: () => void
  categories: Map<number, Category>
  hiddenCategories: Set<number>
  selected: Set<number>
  onToggle: (id: number) => void
}

function tintOf(category: Category): string {
  return category.tint ?? `${category.color}1a`
}

function SentenceMap({ sentences, waiting, onRetry, categories, hiddenCategories, selected, onToggle }: Props) {
  const [highlighted, setHighlighted] = useState(true)

  return (
    <>
      <div className="panel-heading result-heading">
        <div><span className="step">02</span><h2>Sentence map</h2></div>
        <label className="toggle-label">
          <input
            checked={highlighted && sentences !== null}
            disabled={!sentences}
            onChange={(event) => setHighlighted(event.target.checked)}
            type="checkbox"
          />
          <span className="toggle" /> Highlight categories
        </label>
      </div>
      {sentences ? (
        <>
          <p className="selection-hint">
            {selected.size
              ? `${selected.size} ${selected.size === 1 ? 'sentence' : 'sentences'} selected for rewriting. Click again to deselect.`
              : 'Click sentences to select them for rewriting. With none selected, the whole policy is used.'}
          </p>
          <div className="sentence-list">
            {sentences.map((sentence) => {
              const category = sentence.label_id === null ? undefined : categories.get(sentence.label_id)
              const isVisible = highlighted && category !== undefined && !hiddenCategories.has(category.id)
              const isSelected = selected.has(sentence.id)
              return (
                <div
                  aria-pressed={isSelected}
                  className={`sentence ${isVisible ? 'is-highlighted' : ''} ${isSelected ? 'is-selected' : ''}`}
                  key={sentence.id}
                  onClick={() => onToggle(sentence.id)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      onToggle(sentence.id)
                    }
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
            })}
          </div>
        </>
      ) : (
        <div className="empty-state">
          <p>{waiting}</p>
          {onRetry && <button className="secondary-button" onClick={onRetry} type="button">Try again</button>}
        </div>
      )}
    </>
  )
}

export default SentenceMap
