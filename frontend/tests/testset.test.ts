import { describe, expect, it } from 'vitest'
import { firstContextPreview, formatQaCount } from '../src/api/testset'

describe('formatQaCount', () => {
  it('formats total(single/specific/abstract)', () => {
    expect(
      formatQaCount({
        actual_single: 10,
        actual_multi_specific: 10,
        actual_multi_abstract: 10,
        item_count: 30,
      }),
    ).toBe('30(10/10/10)')
    expect(
      formatQaCount({
        actual_single: 0,
        actual_multi_specific: 5,
        actual_multi_abstract: 3,
        item_count: 8,
      }),
    ).toBe('8(0/5/3)')
  })

  it('shows only the total when types are incomplete (imports)', () => {
    // mixed import: some items typed, some not
    expect(
      formatQaCount({
        actual_single: 0,
        actual_multi_specific: 0,
        actual_multi_abstract: 0,
        item_count: 5,
      }),
    ).toBe('5')
    // generated testset with an item deleted afterwards: breakdown is stale
    expect(
      formatQaCount({
        actual_single: 10,
        actual_multi_specific: 10,
        actual_multi_abstract: 10,
        item_count: 29,
      }),
    ).toBe('29')
  })
})

describe('firstContextPreview', () => {
  it('returns first context truncated', () => {
    expect(firstContextPreview(['short text'])).toBe('short text')
    const long = 'x'.repeat(100)
    expect(firstContextPreview([long])).toBe('x'.repeat(40) + '…')
  })

  it('handles empty contexts', () => {
    expect(firstContextPreview([])).toBe('')
  })
})
