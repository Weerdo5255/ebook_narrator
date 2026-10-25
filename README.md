# epub-narrator

Turns a DRM-free EPUB into an illustrated edition plus a chapter-by-chapter audiobook, or into a
comic-book adaptation, using only AI running on your own machine:

- **Ollama** reads the book, finds passages with strong visual description, writes image prompts for
  them, prepares the text for speech, and writes a cover prompt. For the comic it also builds the cast
  of characters, picks each chapter's important plot points and writes a page-by-page comic script.
- **ComfyUI** draws the illustrations, the cover, the character portraits and the comic panels, and
  speaks the narration.
- **ffmpeg** stitches the speech into one audio file per chapter.

Everything ends up in one folder:

```
output/<Book Title>/
  <Book Title>.epub          the original book with illustrations and a new cover
  cover.jpg
  audio/
    01 - Chapter One.mp3     one track per chapter, tagged, with cover art embedded
    02 - Chapter Two.mp3
    ...
    <Book Title>.m3u         playlist
  chapters.json              chapter list, track durations, and every illustration's caption and prompt
  <Book Title>.cbz           the comic (with --comic)
  comic/
    cast/<Name>.jpg          the reference portrait of each character
    comic.json               cast designs (what the book says vs. what was invented) and the full script
```

## 1. Install

**Python 3.10+** and **ffmpeg** (must be on your PATH; `ffmpeg -version` should work).

```bash
pip install -r requirements.txt
```

**Ollama** - install from ollama.com, then pull a model:

```bash
ollama pull llama3.1:8b        # the default; any instruction-tuned model works
```

Bigger models judge scenes better. If your GPU has room, `qwen2.5:14b`, `gemma3:12b` or
`mistral-nemo` are good choices; set `model` in `config.toml` (or pass `--model`). For the comic a
bigger model makes a real difference: it writes better scripts and keeps track of who is who.

**ComfyUI** - install and start it (default address `http://127.0.0.1:8188`). Then:

- *Images:* put an SDXL checkpoint in `ComfyUI/models/checkpoints/` and make sure its file name
  matches `ckpt_name` in `workflows/image_sdxl.json` (default `sd_xl_base_1.0.safetensors`).
  Any SDXL fine-tune works; you can also swap in your own workflow (Flux, SD3, ...) - see section 4.
- *Speech:* install the **comfyui-kokoro** custom node (in ComfyUI Manager search for "Kokoro", or
  `git clone https://github.com/stavsap/comfyui-kokoro` into `ComfyUI/custom_nodes/` and
  `pip install -r requirements.txt` inside it). The model downloads itself on first use.
  The voice is set in `workflows/tts_kokoro.json` (`speaker_name`, default `af_heart`; for a British
  voice such as `bm_george` also set `lang` to `English (British)`).
- *Comic (optional):* install **ComfyUI_IPAdapter_plus** (ComfyUI Manager: "ComfyUI IPAdapter plus",
  or `git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus` into `ComfyUI/custom_nodes/`). It
  lets every panel use the characters' reference portraits. It needs two model files:
  - `ComfyUI/models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`
    (the `image_encoder/model.safetensors` file from the h94/IP-Adapter repository, renamed)
  - `ComfyUI/models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors` (from h94/IP-Adapter,
    `sdxl_models` folder)

  Also set `ckpt_name` in `workflows/comic_panel_sdxl_ipadapter.json` to your SDXL checkpoint. A
  fine-tune with a comic or illustration look gives much better panels than the base model. Without
  IPAdapter you can still make the comic: set `comic.panel_workflow = "workflows/image_sdxl.json"`
  and characters are kept consistent by their fixed descriptions alone.

## 2. Run

```bash
# See how the book was split before spending GPU time
python epub_narrator.py mybook.epub --list-chapters

# Preview which passages would be illustrated (LLM only, no images rendered)
python epub_narrator.py mybook.epub --scenes-only --chapters 1-3

# Try a couple of chapters end to end
python epub_narrator.py mybook.epub --chapters 1-2

# The whole book
python epub_narrator.py mybook.epub

# Comic: check the cast and the script first (LLM only), then draw it
python epub_narrator.py mybook.epub --script-only --chapters 1
python epub_narrator.py mybook.epub --comic --chapters 1-2
python epub_narrator.py mybook.epub --comic
```

Useful options:

| Option | What it does |
|---|---|
| `--chapters 1-3,7` | process only these chapters (numbers from `--list-chapters`) |
| `--no-images` / `--no-audio` / `--no-cover` | skip a part of the job |
| `--no-cleanup` | send the text to TTS as-is (much faster, see below) |
| `--threshold 8` | be pickier about which scenes get an illustration (0-10) |
| `--max-per-chapter 1` | cap illustrations per chapter |
| `--model qwen2.5:14b` | use a different Ollama model |
| `--format m4b` | audio format: mp3, m4a, m4b, opus, flac, wav |
| `--comic` | make the comic (.cbz) instead of the illustrated EPUB and audiobook |
| `--script-only` | plan the comic (cast and page scripts), print it, save `comic/comic.json`, exit |
| `--pages 3` | comic pages per chapter (default: estimated from the chapter's length) |
| `-o DIR`, `-c config.toml` | output folder, config file |

`--comic` makes only the comic. To get the comic as part of every full run alongside the EPUB and
audio, set `enabled = true` under `[comic]` in `config.toml`.

**Stopping and resuming.** Every LLM answer, image and speech chunk is cached in
`output/.cache/<book>/`. Press Ctrl+C at any time and run the same command again: it continues where it
left off, and finished chapter tracks are not re-encoded. Running different `--chapters` ranges on
different days builds up the same book; the EPUB always includes every illustration made so far, and
the comic every chapter whose panels are all drawn. Delete the cache folder when you are happy with
the result.

## 3. How it decides what to illustrate

The chapter is read in passages of about 3,500 characters. For each passage the LLM:

1. lists the concrete visual details the text actually states (setting, light, clothing, actions...),
2. scores the passage 0-10 for how strongly it describes a scene,
3. picks the paragraph where the scene is described best, and writes an image prompt and a caption.

A passage only gets an image when its score is at least `threshold` (default 7) **and** the LLM found
at least `min_visual_details` (default 3) concrete details. The second check stops small models from
giving high scores to passages that are mostly dialogue. Then at most `max_per_chapter` images are kept
per chapter, at least `min_paragraphs_apart` paragraphs apart. The illustration is inserted right after
the chosen paragraph, with the caption under it.

Character appearances the LLM finds along the way (for example "Elara: tall woman, long silver hair,
green cloak") are fed back into later prompts so recurring characters look roughly consistent. Image
prompts describe people by appearance, never by name.

Use `--scenes-only` to check the choices and tune `threshold` before rendering anything.

## 4. The comic edition

`--comic` adapts the book into a graphic novel: a `.cbz` file (a zip of page images that every comic
reader opens, with ComicInfo.xml metadata and a bookmark for each chapter).

**How it is made**

1. *Reading.* The LLM reads the book in passages of about 6,000 characters and notes, for each one,
   who appears (with every detail the text gives about their looks), what happens (each event rated
   from 1, background, to 5, turning point) and where it takes place. By default it reads the whole
   book before anything else (`cast_scope = "book"`), so a character described in chapter 12 looks
   right in chapter 1 too.
2. *Cast.* Name variants are merged ("Lizzy", "Miss Elizabeth" and "Elizabeth Bennet" are one person,
   while "Mr. Bennet" and "Mrs. Bennet" stay two). The characters who appear most get a fixed design:
   every detail the book states is kept, and whatever it doesn't say (age, hair, clothes...) is
   invented to suit the period, the setting and the character's role, and made distinct from the
   rest of the cast. `comic.json` records which traits came from the book and which were invented.
3. *Script.* For each chapter the LLM picks the events that matter to the plot and spreads them over
   the pages (about 0.6 pages per 1,000 words), then writes each page as 4-6 panels: shot type,
   setting, action, who stands where, a narration caption, up to three balloons of dialogue taken from
   the book's own lines wherever possible (speech, thought, shout or whisper), and sound effects.
4. *Portraits.* ComfyUI draws a reference portrait of each cast member. These appear on a "The
   Characters" page after the cover.
5. *Panels.* Each panel is drawn at the shape of its frame on the page. Every character in it is
   described with exactly the same wording each time, and with the IPAdapter workflow their portrait
   steers how they are drawn, applied to their part of the frame (left, centre, right) so two
   characters in one panel don't blend into each other.
6. *Lettering and pages.* The program, not the image model, puts the words in: captions, balloons with
   tails pointing at the speaker, thought clouds, shout bursts, dashed whisper balloons and sound
   effects, in reading order and kept off faces where possible. Panels are laid out in rows (wide
   shots get a row of their own), the first page of each chapter gets a title, and pages are numbered.

**Check before you draw.** `--script-only --chapters 1` runs only the LLM steps and prints the cast
(with each character's full description and which parts were invented) and the script, panel by
panel. It is cheap to repeat with a different `--model` until the adaptation reads well.

**Changing a character.** If a design is wrong, override any part of it in `config.toml`, using the
name the cast list shows:

```toml
[comic.cast."Elizabeth Bennet"]
hair = "dark brown curls pinned up"
outfit = "pale green muslin dress"

[comic.cast."Mr. Darcy"]
image = "refs/darcy.png"     # your own reference picture instead of a generated portrait
```

Fields: `gender`, `age`, `build`, `skin`, `hair`, `eyes`, `face`, `outfit`, `accessories`, `short`
(the few-word nickname used in action descriptions), `look` (replaces the whole description), `seed`
(re-roll the portrait) and `image`. Panels showing that character are redrawn on the next run; all
other cached work is kept. Designs are fixed the first time a character is designed, so they stay the
same across runs; to have the LLM design everyone again, delete `comic/designs.json` in the cache
folder.

**Lettering font.** For a real comic look, use a comic lettering font: drop it into a `fonts/` folder
next to `epub_narrator.py`, or set `comic.font`. The program looks for Anime Ace 2.0 BB
(`animeace2_bld.ttf`), Comic Neue Bold (free from Google Fonts) and Comic Sans Bold, and for titles and
sound effects Bangers (free from Google Fonts) or Impact; otherwise it uses a plain bold sans-serif.

**Other comic settings** in `[comic]`: page size (`page_width`, `page_height`, default 1600x2400),
`panels_per_page`, `style` (the art style added to every panel), `megapixels` (render size; lower it
for SD 1.5), `ref_weight` (how strongly the portraits steer the panels), `uppercase` lettering,
`cover`, `cast_page`, `page_numbers`, and `split = "chapter"` for one `.cbz` per chapter.

## 5. Using your own ComfyUI workflows

Any workflow works, as long as it is exported in **API format** and contains the right placeholders.

1. Build or load the workflow in ComfyUI and check it works.
2. Type the placeholder into the matching text box, e.g. `{{PROMPT}}` in the positive prompt box.
3. Export it with **Workflow > Export (API)** (in older ComfyUI versions, enable Dev mode in the
   settings and use "Save (API Format)").
4. Point `images.workflow`, `cover.workflow`, `narration.workflow`, `comic.panel_workflow` or
   `comic.character_workflow` in `config.toml` at the file.

| Workflow | Required | Optional |
|---|---|---|
| Image / cover / character portrait | `{{PROMPT}}` | `{{NEGATIVE_PROMPT}}`, `{{SEED}}`, `{{WIDTH}}`, `{{HEIGHT}}` |
| Comic panel | `{{PROMPT}}` | the above, plus `{{REF_IMAGE_n}}`, `{{REF_MASK_n}}`, `{{REF_WEIGHT_n}}` (n = 1-4), `{{REF_SHEET}}` |
| Speech (TTS) | `{{TEXT}}` | `{{SEED}}` |

ComfyUI's number boxes cannot hold text, so you don't have to put `{{SEED}}`, `{{WIDTH}}` or
`{{HEIGHT}}` in an image workflow: when they are missing, the program fills any `seed`/`noise_seed`
input, and the `width`/`height` of any "Latent" node, with the right values automatically. Each image
gets a seed derived from its prompt, so re-runs reproduce the same picture.

**Character references in panel workflows.** The program uploads the pictures into ComfyUI's input
folder and fills the placeholders with their names, so they go into **Load Image** (`{{REF_IMAGE_n}}`,
`{{REF_SHEET}}`) and **Load Image (as Mask)** (`{{REF_MASK_n}}`, channel `red`) nodes:

- `{{REF_IMAGE_n}}` - the portrait of the n-th character in the panel, left to right;
- `{{REF_MASK_n}}` - a white-on-black mask of that character's part of the frame (the whole frame
  when they are alone), for IPAdapter's `attn_mask`;
- `{{REF_WEIGHT_n}}` - the strength to use (`ref_weight` / `ref_weight_multi`). It is a number, so put
  it in by editing the exported JSON (`"weight": "{{REF_WEIGHT_1}}"`), as the shipped workflow does.
- Slots the panel does not need get a grey placeholder picture, an all-black mask and weight 0, which
  switches them off; panels without known characters are drawn from the prompt alone.
- `{{REF_SHEET}}` - all the panel's character portraits side by side in one picture, for models that
  take one reference image (Flux Kontext, Qwen-Image-Edit and similar edit models).

`workflows/comic_panel_sdxl_ipadapter.json` uses two slots; add slots 3 and 4 by copying its three
nodes per character if you often have crowded panels.

The workflow needs a Save Image node (images) or a Save Audio node (speech). The program sets every
`filename_prefix`, so files land in `ComfyUI/output/epub_narrator/<book>/`.

Other TTS nodes (F5-TTS, Chatterbox, VibeVoice, etc.) work the same way: put `{{TEXT}}` in the text
input and keep the voice settings inside the workflow. If a model gets worse with long input, lower
`narration.chunk_chars`. If it clones a voice from a reference clip, keep the clip and the seed fixed so
every chapter sounds the same.

## 6. What happens to the text before it is spoken

- Footnotes, footnote markers, page numbers and the table of contents are not read.
- Scene breaks (`<hr>`, `* * *`) become a longer pause; headings get a pause after them;
  "CHAPTER IV." is spoken as "Chapter 4".
- With `llm_cleanup` on (default), Ollama spells out numbers, dates and abbreviations
  ("Dr." becomes "Doctor", "1820" becomes "eighteen twenty"). Each paragraph it returns is checked
  against the original; if the length changes too much the original wording is kept, so the LLM cannot
  summarise or rewrite the book.
- Text is sent to TTS in chunks of up to `chunk_chars`, split at sentence boundaries, and the chunks
  are joined with the configured pauses. Loudness is normalised per chapter.

## 7. Time and GPU memory

Rough guide for a 100,000-word novel on a single consumer GPU:

- scene analysis: ~170 LLM requests (one per ~3,500-character passage)
- text cleanup: ~250 LLM requests (the slowest LLM part; `--no-cleanup` skips it)
- illustrations: a few dozen images
- narration: about 11 hours of audio. Kokoro is fast; voice-cloning models can take far longer.
- comic: ~100 LLM requests to read the book, one per chapter to plan it and one per page to script it;
  about 60 pages, so 300+ panels plus a dozen portraits. At 10-20 seconds per SDXL panel that is one
  to two hours of drawing; try a chapter or two first.

The work runs in phases (all LLM work, then images, then speech) and the LLM is unloaded before
ComfyUI starts, and ComfyUI's image model is unloaded before speech, so an 8-12 GB GPU is enough
(SDXL with IPAdapter fits in 10-12 GB). Turn this off with `ollama.unload_when_done` /
`comfyui.free_memory_between_phases` if you have plenty of VRAM.

## 8. Troubleshooting

- **"Cannot reach Ollama/ComfyUI"** - start them, or fix the `url` in `config.toml`.
- **"model ... is not installed"** - `ollama pull <model>`.
- **"was saved in ComfyUI's normal (UI) format"** - re-export with Export (API).
- **"uses node(s) this ComfyUI does not have"** - install the custom node pack it names and restart
  ComfyUI (for the comic, or switch `comic.panel_workflow` to `workflows/image_sdxl.json`).
- **"ComfyUI rejected the workflow: ... ckpt_name ..."** - the checkpoint file name in the workflow
  does not match a file in `models/checkpoints/`.
- **"ClipVision model not found" / "IPAdapter model not found"** - the IPAdapter model files are
  missing or misnamed; see section 1.
- **Wrong chapter split** - check `--list-chapters`. Sections shorter than `chapters.min_chars` and
  titles matching `chapters.skip_titles` are skipped; set `use_toc = false` to treat every file in the
  book as its own chapter.
- **Too many / too few illustrations** - adjust `threshold`, `min_visual_details`, `max_per_chapter`,
  and check with `--scenes-only`. A bigger Ollama model makes better judgements.
- **A chapter's narration failed** - the run finishes the other chapters and reports which ones
  failed; run again to retry them. `-v` shows details.
- **Comic panels failed** - they show as grey boxes and the run says which chapters; run again to
  redraw just those panels.
- **Characters look alike or change between panels** - give them more distinct designs in
  `[comic.cast]`, raise `ref_weight` a little, or use an SDXL fine-tune that follows prompts better.
  If two characters blend in shared panels, lower `ref_weight_multi`.
- **Panels contain garbled writing** - image models can't write; keep "text, speech bubble" in
  `comic.negative_prompt`. All real lettering is added afterwards.
- **"is DRM-protected"** - only DRM-free EPUBs can be read.

## Notes

- The original book is never modified; a new EPUB is written next to the audio.
- If the book already has a cover, the new one replaces it at the same size and position, so every
  reading app shows it; `cover.keep_original = true` (or `--no-cover`) keeps the old one.
- Illustrations are ordinary images in the EPUB, so they show up in any reader. The audio is separate
  files rather than EPUB "read-aloud" media overlays.
- The comic is an adaptation: it condenses each chapter to its main events, and invented character
  details are only as good as the model's guesses. `comic/comic.json` shows which events each page
  covers, every panel's script and image prompt, and which character details were invented.
