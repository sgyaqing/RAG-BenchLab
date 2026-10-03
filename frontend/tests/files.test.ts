import { describe, expect, it } from 'vitest'
import { analyzePaths, isAllowedUpload, isSupportedFile, isZipFile } from '../src/utils/files'

describe('isSupportedFile', () => {
  it('accepts supported extensions case-insensitively', () => {
    expect(isSupportedFile('a.txt')).toBe(true)
    expect(isSupportedFile('b.PDF')).toBe(true)
    expect(isSupportedFile('c.docx')).toBe(true)
  })

  it('rejects unsupported files including zip', () => {
    expect(isSupportedFile('a.zip')).toBe(false)
    expect(isSupportedFile('a.exe')).toBe(false)
    expect(isSupportedFile('noext')).toBe(false)
  })
})

describe('analyzePaths', () => {
  it('empty selection', () => {
    expect(analyzePaths([])).toBe('empty')
  })

  it('single zip', () => {
    expect(analyzePaths(['pack.zip'])).toBe('zip')
  })

  it('multiple zips are invalid', () => {
    expect(analyzePaths(['a.zip', 'b.zip'])).toBe('invalid')
  })

  it('one or more loose supported files', () => {
    expect(analyzePaths(['readme.md'])).toBe('files')
    expect(analyzePaths(['a.txt', 'b.pdf', 'c.docx'])).toBe('files')
  })

  it('loose unsupported files are invalid', () => {
    expect(analyzePaths(['a.txt', 'b.exe'])).toBe('invalid')
  })

  it('directory selection (all paths under one top-level dir)', () => {
    expect(analyzePaths(['dir/a.txt', 'dir/sub/b.pdf'])).toBe('dir')
    expect(analyzePaths(['dir/a.txt'])).toBe('dir')
  })

  it('two directories are invalid', () => {
    expect(analyzePaths(['dir1/a.txt', 'dir2/b.txt'])).toBe('invalid')
  })

  it('mixing loose files and directory is invalid', () => {
    expect(analyzePaths(['a.txt', 'dir/b.txt'])).toBe('invalid')
  })

  it('mixing zip and files is invalid', () => {
    expect(analyzePaths(['a.zip', 'b.txt'])).toBe('invalid')
  })
})

describe('isZipFile', () => {
  it('detects zip extension', () => {
    expect(isZipFile('a.zip')).toBe(true)
    expect(isZipFile('a.ZIP')).toBe(true)
    expect(isZipFile('a.txt')).toBe(false)
  })
})

describe('isAllowedUpload', () => {
  it('allows supported files at any depth', () => {
    expect(isAllowedUpload('a.txt')).toBe(true)
    expect(isAllowedUpload('dir/sub/b.pdf')).toBe(true)
  })

  it('allows zip only at the top level', () => {
    expect(isAllowedUpload('pack.zip')).toBe(true)
    expect(isAllowedUpload('dir/pack.zip')).toBe(false)
  })

  it('rejects unsupported files', () => {
    expect(isAllowedUpload('a.exe')).toBe(false)
    expect(isAllowedUpload('dir/a.exe')).toBe(false)
  })
})
