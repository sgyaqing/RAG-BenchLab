import { withRelativePath } from './files'

/**
 * The files a drag-and-drop handed over, folders included.
 *
 * A drop is the one gesture that takes all three things this dialog accepts —
 * loose files, a zip and a folder — but the browser does not say which one it
 * was: `dataTransfer.files` flattens a dropped folder into loose files with no
 * path, and a path is exactly how the rest of the pipeline tells a folder from
 * loose files. So the entries are read one by one here instead: a file stays a
 * file, a directory is walked and every file under it comes back carrying the
 * relative path a `webkitdirectory` input would have given it.
 *
 * Falls back to `dataTransfer.files` when the entries cannot be read — a
 * browser without `webkitGetAsEntry`, or a drop we could not open. That is the
 * old behaviour: loose files still work, a folder arrives as loose files.
 */

/** The parts of a FileSystemEntry this module walks. */
type Entry = {
  isFile: boolean
  isDirectory: boolean
  name: string
  file(onOk: (f: File) => void, onErr?: (e: unknown) => void): void
  createReader(): {
    // Batched by spec: called repeatedly until it returns an empty list.
    readEntries(onOk: (entries: Entry[]) => void, onErr?: (e: unknown) => void): void
  }
}

function toPromise<T>(fn: (ok: (v: T) => void, err?: (e: unknown) => void) => void): Promise<T> {
  return new Promise((resolve, reject) => fn(resolve, reject))
}

async function readAll(entry: Entry): Promise<Entry[]> {
  const reader = entry.createReader()
  const out: Entry[] = []
  for (;;) {
    const batch = await toPromise<Entry[]>((ok, err) => reader.readEntries(ok, err))
    if (batch.length === 0) return out
    out.push(...batch)
  }
}

async function walk(dir: Entry, prefix: string, out: File[]): Promise<void> {
  for (const entry of await readAll(dir)) {
    const path = `${prefix}/${entry.name}`
    if (entry.isFile) {
      const file = await toPromise<File>((ok, err) => entry.file(ok, err))
      out.push(withRelativePath(file, path))
    } else if (entry.isDirectory) {
      await walk(entry, path, out)
    }
  }
}

export async function filesFromDataTransfer(dt: DataTransfer): Promise<File[]> {
  const items = Array.from(dt.items ?? []).filter((item) => item.kind === 'file')
  const entries: Entry[] = []
  for (const item of items) {
    // Not on every browser: absent means "use dataTransfer.files", below.
    const get = (item as DataTransferItem & { webkitGetAsEntry?: () => Entry | null })
      .webkitGetAsEntry
    if (typeof get === 'function') {
      const entry = get.call(item)
      if (entry) entries.push(entry)
    }
  }

  const out: File[] = []
  for (const entry of entries) {
    if (entry.isFile) {
      const file = await toPromise<File>((ok, err) => entry.file(ok, err))
      out.push(withRelativePath(file, file.name))
    } else if (entry.isDirectory) {
      await walk(entry, entry.name, out)
    }
  }
  if (out.length > 0) return out
  // Everything here comes back carrying a path, so the caller never has to
  // know which route a file took. A browser that filled webkitRelativePath in
  // itself keeps it — the bare name is only the fallback for the rest.
  return Array.from(dt.files ?? []).map((file) =>
    file.webkitRelativePath ? file : withRelativePath(file, file.name),
  )
}
