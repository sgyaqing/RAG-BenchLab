import { afterEach, describe, expect, it } from 'vitest'

import { canPickDirectory, pickDirectory } from '../src/utils/directory'

type FakeNode = { name: string; kind: 'file' | 'directory'; children?: FakeNode[] }

function handle(node: FakeNode) {
  return {
    name: node.name,
    kind: node.kind,
    async getFile() {
      return new File(['x'], node.name)
    },
    async *values() {
      for (const child of node.children ?? []) yield handle(child)
    },
  }
}

function installPicker(picker: unknown) {
  ;(globalThis as unknown as { window: unknown }).window = { showDirectoryPicker: picker }
}

const TREE: FakeNode = {
  name: 'corpus',
  kind: 'directory',
  children: [
    { name: 'a.md', kind: 'file' },
    { name: 'skip.exe', kind: 'file' },
    {
      name: 'nested',
      kind: 'directory',
      children: [{ name: 'b.pdf', kind: 'file' }],
    },
  ],
}

afterEach(() => {
  delete (globalThis as unknown as { window?: unknown }).window
})

describe('directory picking', () => {
  it('is unavailable where the browser has no chooser', () => {
    expect(canPickDirectory()).toBe(false)
  })

  it('reports itself available where the browser has one', () => {
    installPicker(async () => handle(TREE))
    expect(canPickDirectory()).toBe(true)
  })

  it('walks the folder and gives every file the path a directory input would', async () => {
    installPicker(async () => handle(TREE))
    const files = await pickDirectory()

    expect(files.map((f) => f.webkitRelativePath)).toEqual([
      'corpus/a.md',
      'corpus/skip.exe',
      'corpus/nested/b.pdf',
    ])
    expect(files.map((f) => f.name)).toEqual(['a.md', 'skip.exe', 'b.pdf'])
  })

  it('returns nothing when the user cancels the chooser', async () => {
    installPicker(async () => {
      throw new DOMException('cancelled', 'AbortError')
    })
    expect(await pickDirectory()).toEqual([])
  })

  it('returns nothing when there is no chooser at all', async () => {
    expect(await pickDirectory()).toEqual([])
  })
})
