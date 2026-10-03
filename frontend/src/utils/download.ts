/** Download `url`, preferring a native "Save As" dialog (File System Access
 *  API, Chrome/Edge) so the user can rename the file and pick the location.
 *  Falls back to a plain anchor download in browsers without the API.
 *  Resolves quietly when the user cancels the dialog; throws on real errors.
 *  Note: showSaveFilePicker must be the first await in a click handler to
 *  keep transient user activation, so the fetch happens after picking. */
export async function downloadWithSaveAs(
  url: string,
  suggestedName: string,
  accept: Record<string, string[]>,
): Promise<void> {
  if (window.showSaveFilePicker) {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName,
        types: [{ description: suggestedName, accept }],
      })
      const resp = await fetch(url)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const writable = await handle.createWritable()
      await writable.write(await resp.blob())
      await writable.close()
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return // user cancelled
      throw e
    }
    return
  }
  const a = document.createElement('a')
  a.href = url
  a.download = suggestedName
  a.click()
}
