import { describe, expect, it } from 'vitest'

import { filesFromDataTransfer } from '../src/utils/drop'
import { analyzePaths } from '../src/utils/files'

type Node = { name: string; kind: 'file' | 'directory'; children?: Node[] }

type FakeEntry = {
  isFile: boolean
  isDirectory: boolean
  name: string
  file(ok: (f: File) => void): void
  createReader(): { readEntries(ok: (entries: FakeEntry[]) => void): void }
}

function entry(node: Node): FakeEntry {
  // One child per readEntries call, then an empty list — the real reader
  // returns a batch per call and a folder is only finished when it says so.
  const batches = (node.children ?? []).map((child) => [entry(child)])
  return {
    isFile: node.kind === 'file',
    isDirectory: node.kind === 'directory',
    name: node.name,
    file(ok) {
      ok(new File(['x'], node.name))
    },
    createReader() {
      let call = 0
      return {
        readEntries(ok) {
          ok(batches[call++] ?? [])
        },
      }
    },
  }
}

/** A DataTransfer carrying dropped entries; `noEntries` drops the API entirely
 *  the way a browser without webkitGetAsEntry does. */
function drop(
  nodes: Node[],
  { files = [] as File[], noEntries = false, unreadable = false } = {},
): DataTransfer {
  return {
    items: noEntries
      ? files.map(() => ({ kind: 'file' }))
      : nodes.map((node) => ({
          kind: 'file',
          webkitGetAsEntry: () => (unreadable ? null : entry(node)),
        })),
    files,
  } as unknown as DataTransfer
}

describe('files dropped on the corpus dialog', () => {
  it('walks a dropped folder and keeps the paths a directory input would give', async () => {
    const files = await filesFromDataTransfer(
      drop([
        {
          name: 'corpus',
          kind: 'directory',
          children: [
            { name: 'a.md', kind: 'file' },
            { name: 'skip.exe', kind: 'file' },
            { name: 'nested', kind: 'directory', children: [{ name: 'b.pdf', kind: 'file' }] },
          ],
        },
      ]),
    )

    expect(files.map((f) => f.webkitRelativePath)).toEqual([
      'corpus/a.md',
      'corpus/skip.exe',
      'corpus/nested/b.pdf',
    ])
    // The whole point of the paths: this is what tells a folder from loose files.
    expect(analyzePaths(files.map((f) => f.webkitRelativePath))).toBe('dir')
  })

  it('leaves a dropped zip loose, so it still counts as one zip', async () => {
    const files = await filesFromDataTransfer(drop([{ name: 'docs.zip', kind: 'file' }]))

    expect(files.map((f) => f.webkitRelativePath)).toEqual(['docs.zip'])
    expect(analyzePaths(files.map((f) => f.webkitRelativePath))).toBe('zip')
  })

  it('keeps a folder and a loose zip apart, which the dialog rejects as mixed', async () => {
    const files = await filesFromDataTransfer(
      drop([
        { name: 'docs.zip', kind: 'file' },
        { name: 'corpus', kind: 'directory', children: [{ name: 'a.md', kind: 'file' }] },
      ]),
    )

    expect(analyzePaths(files.map((f) => f.webkitRelativePath))).toBe('invalid')
  })

  it('falls back to dataTransfer.files where the entry API is missing', async () => {
    const loose = new File(['x'], 'a.md')
    const files = await filesFromDataTransfer(drop([], { files: [loose], noEntries: true }))

    expect(files.map((f) => f.webkitRelativePath)).toEqual(['a.md'])
  })

  it('falls back when an entry cannot be read', async () => {
    const loose = new File(['x'], 'a.md')
    const files = await filesFromDataTransfer(
      drop([{ name: 'a.md', kind: 'file' }], { files: [loose], unreadable: true }),
    )

    expect(files.map((f) => f.webkitRelativePath)).toEqual(['a.md'])
  })
})
