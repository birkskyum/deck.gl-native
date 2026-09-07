// Copies the repository's docs into the site: every docs/*.md becomes a page, and the
// pictures they use become static assets. Keeping one copy of the prose in the repository
// means the site never drifts from it.
import { cp, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { base } from '../base.mjs'

const here = dirname(fileURLToPath(import.meta.url))
const repo = join(here, '..', '..')
const docs = join(repo, 'docs')
const out = join(here, '..', 'src', 'content', 'docs', 'docs')
const images = join(here, '..', 'public', 'images')
// Starlight optimizes the hero picture, so that one has to live under src/ as an asset
const assets = join(here, '..', 'src', 'assets')
const hero = 'texture-render.png'

/** The order pages appear in the sidebar, and the label each one gets. */
const pages = {
  'rust-port.md': { title: 'Design notes', order: 1 },
  'json.md': { title: 'JSON layers', order: 2 },
  'extensions.md': { title: 'Extensions', order: 3 },
  'benchmarks.md': { title: 'Benchmarks', order: 4 },
  'maplibre-native.md': { title: 'maplibre-native', order: 5 },
  'cpp-prototype.md': { title: 'The C++ prototype', order: 6 },
}

await rm(out, { recursive: true, force: true })
await mkdir(out, { recursive: true })
await rm(images, { recursive: true, force: true })
await cp(join(docs, 'images'), images, { recursive: true })
await mkdir(assets, { recursive: true })
await cp(join(docs, 'images', hero), join(assets, hero))

for (const file of await readdir(docs)) {
  if (!file.endsWith('.md')) continue
  const page = pages[file]
  if (!page) {
    console.warn(`docs/${file} has no entry in sync-docs.mjs, skipping it`)
    continue
  }
  const source = await readFile(join(docs, file), 'utf8')
  const body = source
    // The first heading becomes the page title, which Starlight renders itself
    .replace(/^#\s+.*\n+/, '')
    // Pictures live under /images on the site
    .replace(/\]\(images\//g, `](${base}/images/`)
    // Links between documents keep working
    .replace(/\]\(([a-z0-9-]+)\.md\)/g, `](${base}/docs/$1/)`)
    .replace(/\]\(\.\.\/([A-Za-z0-9_.-]+)\)/g, '](https://github.com/birkskyum/deck.gl-native/blob/main/$1)')
  const name = file.replace(/\.md$/, '')
  // The page is a copy, so send the edit link to the document it was copied from
  const editUrl = `https://github.com/birkskyum/deck.gl-native/edit/main/docs/${file}`
  const frontmatter =
    `---\ntitle: ${page.title}\neditUrl: ${editUrl}\nsidebar:\n  order: ${page.order}\n---\n\n`
  await writeFile(join(out, `${name}.md`), frontmatter + body)
}

console.log(`synced ${Object.keys(pages).length} pages and the pictures from docs/`)
