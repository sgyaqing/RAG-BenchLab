/**
 * Picking a folder the browser will hand back as the result.
 *
 * `webkitdirectory` is the only folder-picking API Safari and Firefox have, but
 * on macOS the dialog Chrome opens for it is the plain file panel: a folder is
 * somewhere to walk into, never something to select, so pressing Open descends
 * into it — and once inside, with nothing selected, Open greys out. There is no
 * way to choose a folder with it there.
 *
 * Chrome and Edge also expose showDirectoryPicker(), which opens the system's
 * folder chooser and returns the folder itself. So those two use it, and every
 * other browser keeps the input element that already works for it — Safari's
 * panel does return the folder, its button even relabels to "Upload".
 *
 * Absent outside a secure context: served over plain http on a LAN address the
 * function is undefined, and the old path is used, which is exactly what
 * happened before this file existed.
 */

import { withRelativePath } from './files'

/** The parts of a FileSystemDirectoryHandle this module walks. */
type DirectoryHandle = {
  name: string
  kind: 'file' | 'directory'
  values(): AsyncIterableIterator<DirectoryHandle>
  getFile(): Promise<File>
}

type PickerWindow = { showDirectoryPicker?: () => Promise<DirectoryHandle> }

function pickerWindow(): PickerWindow | null {
  return typeof window === 'undefined' ? null : (window as unknown as PickerWindow)
}

export function canPickDirectory(): boolean {
  return typeof pickerWindow()?.showDirectoryPicker === 'function'
}

/**
 * The chosen folder's files, each carrying the same relative path a
 * `webkitdirectory` input would have given it.
 *
 * Returns an empty list when the chooser is unavailable or the user cancels.
 * The caller does not need to tell those apart — either way it does nothing,
 * and popping the old dialog after a cancel would only ask twice.
 */
export async function pickDirectory(): Promise<File[]> {
  const picker = pickerWindow()?.showDirectoryPicker
  if (!picker) return []

  let root: DirectoryHandle
  try {
    root = await picker()
  } catch {
    return []
  }

  const files: File[] = []
  await collect(root, root.name, files)
  return files
}

async function collect(dir: DirectoryHandle, prefix: string, out: File[]): Promise<void> {
  for await (const entry of dir.values()) {
    const path = `${prefix}/${entry.name}`
    if (entry.kind === 'file') {
      const file = await entry.getFile()
      out.push(withRelativePath(file, path))
    } else {
      await collect(entry, path, out)
    }
  }
}
