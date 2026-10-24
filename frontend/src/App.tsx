import { useMemo, useState } from 'react'
import './App.css'

type Category = { name: string; color: string; tint: string }

const categories: Category[] = [
  { name: 'First Party Collection/Use', color: '#e8754e', tint: '#fff0ea' },
  { name: 'Third Party Sharing/Collection', color: '#6ea84f', tint: '#eef7e8' },
  { name: 'Other', color: '#b39a75', tint: '#f5f0e9' },
  { name: 'User Choice/Control', color: '#507bc7', tint: '#edf3ff' },
  { name: 'Do Not Track', color: '#9367a8', tint: '#f4ecf8' },
  { name: 'International and Specific Audiences', color: '#dd657c', tint: '#fff0f2' },
  { name: 'Data Security', color: '#3aab9c', tint: '#e9f8f5' },
  { name: 'Policy Change', color: '#a57bc1', tint: '#f5eff9' },
  { name: 'Data Retention', color: '#e3a33b', tint: '#fff6df' },
  { name: 'User Access, Edit and Deletion', color: '#31a8b5', tint: '#e7f7f8' },
]

const examplePolicy = `At BrightCart, your privacy is important to us. When you sign up, place an order, or contact our support team, we collect personal information such as your name, email address, shipping details, and payment method to process your requests and improve your shopping experience.

We may share certain information, such as your email address, with trusted third-party marketing partners who assist us in delivering personalized promotions. These partners are required to maintain confidentiality and are not permitted to use your data for other purposes.

You have the ability to manage your communication preferences at any time. To stop receiving promotional emails, simply click the unsubscribe link in any message or update your account settings.`

const categoryForSentence = [0, 0, 1, 1, 3, 3]

function App() {
  const [policy, setPolicy] = useState(examplePolicy)
  const [analyzed, setAnalyzed] = useState(true)
  const [highlighted, setHighlighted] = useState(true)
  const [activeCategories, setActiveCategories] = useState(() => new Set(categories.map(({ name }) => name)))
  const [rewriteMode, setRewriteMode] = useState('Simplify for Adults')
  const sentences = useMemo(() => policy.match(/[^.!?]+[.!?]+/g)?.map((text) => text.trim()) ?? [], [policy])
  const categoryCounts = categories.map((category, index) => ({ ...category, count: categoryForSentence.filter((categoryIndex) => categoryIndex === index).length }))

  function toggleCategory(name: string) {
    setActiveCategories((current) => {
      const next = new Set(current)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  return (
    <div className="app-shell">
      <aside className="category-rail">
        <div className="rail-label">Categories</div>
        <div className="category-list">
          {categoryCounts.map((category) => <label className="category-chip" key={category.name} style={{ backgroundColor: category.color }}><input checked={activeCategories.has(category.name)} onChange={() => toggleCategory(category.name)} type="checkbox" /><span>{category.name}</span><strong>{category.count}</strong></label>)}
        </div>
        <div className="rail-footnote">Local analysis<br /><span>Nothing leaves this device</span></div>
      </aside>

      <main className="workspace">
        <header className="topbar"><div><p className="eyebrow">Privacy Policy Reader</p><h1>See what a policy really says.</h1></div><span className="offline-badge"><i /> Offline mode</span></header>

        <section className="intro-panel">
          <div className="panel-heading"><div><span className="step">01</span><h2>Bring in a policy</h2></div><span className="status-dot">{analyzed ? 'Analyzed locally' : 'Ready to analyze'}</span></div>
          <textarea aria-label="Privacy policy text" value={policy} onChange={(event) => { setPolicy(event.target.value); setAnalyzed(false) }} />
          <div className="action-row"><span className="character-count">{policy.length.toLocaleString()} characters · {sentences.length} sentences</span><button className="primary-button" onClick={() => { setAnalyzed(true); setHighlighted(true) }} type="button"><span>Analyze categories</span><b>→</b></button></div>
        </section>

        <div className="content-grid">
          <section className="results-panel">
            <div className="panel-heading result-heading"><div><span className="step">02</span><h2>Sentence map</h2></div><label className="toggle-label"><input checked={highlighted && analyzed} disabled={!analyzed} onChange={(event) => setHighlighted(event.target.checked)} type="checkbox" /><span className="toggle" /> Highlight categories</label></div>
            <div className="sentence-list">
              {sentences.map((sentence, index) => { const category = categories[categoryForSentence[index] ?? index % categories.length]; const isVisible = highlighted && activeCategories.has(category.name) && analyzed; return <div className={`sentence ${isVisible ? 'is-highlighted' : ''}`} key={`${sentence}-${index}`} style={{ '--sentence-tint': category.tint, '--sentence-color': category.color } as React.CSSProperties}><span className="sentence-number">{String(index + 1).padStart(2, '0')}</span><p>{sentence}</p>{analyzed && <span className="sentence-tag">{category.name}</span>}</div> })}
              {!sentences.length && <div className="empty-state">Paste policy text above to see sentence-level results.</div>}
            </div>
          </section>

          <section className="rewrite-panel"><div className="panel-heading"><div><span className="step">03</span><h2>Make it clearer</h2></div><span className="spark">✦</span></div><p className="rewrite-copy">Choose a reading style. The rewrite keeps the original details and stays on your device.</p><div className="rewrite-options">{['Simplify for Kids', 'Simplify for Adults', 'Paraphrase'].map((mode) => <button className={rewriteMode === mode ? 'selected' : ''} onClick={() => setRewriteMode(mode)} key={mode} type="button"><span>{mode === 'Paraphrase' ? '↗' : mode === 'Simplify for Kids' ? '☼' : '◌'}</span>{mode}</button>)}</div><div className="rewrite-result"><span className="result-label">{rewriteMode}</span><p>BrightCart uses your information to provide its services and make your shopping experience better. It may share your email with trusted partners who help send you relevant offers.</p><span className="source-link">Based on sentences 1–4 ↗</span></div></section>
        </div>
      </main>
    </div>
  )
}

export default App
