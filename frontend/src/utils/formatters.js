export function formatTime(timeStr) {
  if (!timeStr) return ''
  const date = new Date(timeStr)
  return date.toLocaleDateString('zh-CN', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

export function formatVolume(vol) {
  if (!vol) return '-'
  if (vol >= 100000000) {
    return (vol / 100000000).toFixed(2) + '万亿'
  }
  return (vol / 10000).toFixed(1) + '亿'
}

export function formatMoney(val) {
  if (!val && val !== 0) return '-'
  const sign = val > 0 ? '+' : ''
  if (Math.abs(val) >= 100000000) {
    return sign + (val / 100000000).toFixed(2) + '亿'
  }
  if (Math.abs(val) >= 10000) {
    return sign + (val / 10000).toFixed(1) + '万'
  }
  return sign + val.toFixed(0)
}

export function formatChange(val) {
  if (!val && val !== 0) return '-'
  const sign = val > 0 ? '+' : ''
  return sign + val.toFixed(2) + '%'
}

export function formatPrice(price) {
  if (!price && price !== 0) return '-'
  return price.toFixed(2)
}

export function getCategoryClass(category) {
  const map = {
    '宏观': 'macro',
    '政策': 'policy',
    '行业': 'industry',
    '全球': 'global',
  }
  return map[category] || ''
}

export function getSentimentLabel(sentiment) {
  if (!sentiment) return '中性'
  const up = sentiment.up_count || 0
  const down = sentiment.down_count || 0
  if (up > down * 2) return '偏热'
  if (down > up * 2) return '偏冷'
  return '中性'
}

export function getSentimentType(sentiment) {
  if (!sentiment) return 'neutral'
  const up = sentiment.up_count || 0
  const down = sentiment.down_count || 0
  if (up > down * 2) return 'hot'
  if (down > up * 2) return 'cold'
  return 'neutral'
}

export function getSentimentText(sentiment) {
  const map = { positive: '😊 积极', negative: '😟 消极', neutral: '😐 中性' }
  return map[sentiment] || '😐 中性'
}

export function getImpactLabel(impact) {
  const map = { bullish: '看涨', bearish: '看跌', neutral: '中性' }
  return map[impact] || '中性'
}

export function getMoneyClass(val) {
  if (!val) return ''
  return val > 0 ? 'up' : 'down'
}

export function truncateContent(content, maxLength = 100) {
  if (!content) return ''
  return content.length > maxLength ? content.substring(0, maxLength) + '...' : content
}

export function sanitizeHtml(html) {
  const doc = new DOMParser().parseFromString(html, 'text/html')
  doc.querySelectorAll('script, iframe, object, embed, form').forEach(el => el.remove())
  doc.querySelectorAll('*').forEach(el => {
    for (const attr of [...el.attributes]) {
      if (attr.name.startsWith('on') || attr.name === 'srcdoc') {
        el.removeAttribute(attr.name)
      }
      if (['href', 'src', 'action'].includes(attr.name) &&
          attr.value.trim().toLowerCase().startsWith('javascript:')) {
        el.removeAttribute(attr.name)
      }
    }
  })
  return doc.body.innerHTML
}
