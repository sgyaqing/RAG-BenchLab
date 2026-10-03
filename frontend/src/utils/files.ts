export const SUPPORTED_EXTENSIONS = ['.txt', '.md', '.pdf', '.docx', '.pptx', '.xlsx', '.xls']

export function extOf(name: string): string {
  const i = name.lastIndexOf('.')
  return i >= 0 ? name.slice(i).toLowerCase() : ''
}

export function isSupportedFile(name: string): boolean {
  return SUPPORTED_EXTENSIONS.includes(extOf(name))
}

export function isZipFile(name: string): boolean {
  return extOf(name) === '.zip'
}

/**
 * Uploadable: supported files anywhere; zip only as a standalone top-level
 * file (zips inside a directory are skipped, per design — no nested unzip).
 */
export function isAllowedUpload(path: string): boolean {
  if (isSupportedFile(path)) return true
  return isZipFile(path) && !path.includes('/')
}

export type SelectionMode = 'empty' | 'zip' | 'dir' | 'files' | 'invalid'

/**
 * Give a File the relative path a `webkitdirectory` input would have handed it.
 *
 * A file that arrives through a directory picker or a folder drop carries no
 * `webkitRelativePath` of its own — that getter is only populated for the
 * input it was invented for. An own property shadows it, so everything
 * downstream (the zip/directory/file classification above, the upload's path
 * field) sees the same shape whichever way the file got here.
 */
export function withRelativePath(file: File, path: string): File {
  Object.defineProperty(file, 'webkitRelativePath', { value: path, configurable: true })
  return file
}

/** Top-level directory name of a nested path, null for loose files. */
export function topDir(path: string): string | null {
  const i = path.indexOf('/')
  return i > 0 ? path.slice(0, i) : null
}

/**
 * Classify the current selection: a single zip, a single directory
 * (all paths nested under the same top-level directory), or one or more
 * loose supported files. Mixed combinations are invalid.
 */
export function analyzePaths(paths: string[]): SelectionMode {
  if (paths.length === 0) return 'empty'

  const nested = paths.filter((p) => topDir(p) !== null)
  const loose = paths.filter((p) => topDir(p) === null)
  if (nested.length > 0 && loose.length > 0) return 'invalid'

  if (nested.length > 0) {
    const tops = new Set(nested.map((p) => topDir(p)))
    return tops.size === 1 ? 'dir' : 'invalid'
  }

  if (paths.some(isZipFile)) {
    return paths.length === 1 ? 'zip' : 'invalid'
  }
  return paths.every(isSupportedFile) ? 'files' : 'invalid'
}
