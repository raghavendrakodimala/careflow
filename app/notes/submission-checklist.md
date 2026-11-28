# Final project submission: scratchpad

Source: email from the program team (pasted 2026-11-15). Edit this file as items get done.

## Key points
- **Deadline: Wednesday 25 November 2026, 11:59 PM IST** (10 days after 15 Nov).
- One submission link (in the email). Teams of two: **only one person submits**, with the partner's full name and email ready.
- Three things to submit: code, presentation (PDF), video demo.

## To-do

### 1. Code (pick one)
- [ ] **GitHub:** public repo link, or private with view access for the `ailaunchpad-dev` account.
- [ ] **Colab:** Share -> "Anyone with the link" as Viewer; run Runtime > Run all first to confirm it works end to end.
- [ ] **Local code:** upload files incl. `requirements.txt` to a Google Drive folder named `[Problem Statement] - [Your Name / Team Names]` (e.g. `CareFlow - <name>`), shared "Anyone with the link" as Viewer, or with admin.ailaunchpad@cse.iith.ac.in.

### 2. Presentation and video
- [ ] Slide deck exported as **PDF**.
- [ ] Video demo, **10 minutes max**: explain the presentation and demo the working system.
- [ ] Upload video to Google Drive or YouTube (Unlisted); access "Anyone with the link can view".

### 3. Pre-submit checks
- [ ] Code and video links are "Anyone with the link can view".
- [x] Code does **not rely on local file paths** (checked 2026-11-23: paths are built relative to the files; no user or drive paths in any `.py`).
- [ ] Presentation is a PDF; video is <= 10 minutes.
- [ ] (Pair only) partner's full name and email ready.

## Project-specific notes
- Do not commit `.env` or `data/` (toolkit ground rule). If the repo is public, check no API key is in any file or history.
- Current `app` reads the data and key from `../capstone-data-toolkit/` via paths relative to the file (not absolute), but it still expects the toolkit's `.env` and generated `data/` to exist. **Done:** the data is committed, and the root `README.md` gives setup and run instructions (2026-11-23).
- Dates differ: the project brief says final presentations are 15-16 August, but this email's deadline is 25 Nov 2026. Treat the email as current; confirm with the program team if unsure.
- Milestone status (2026-11-23): M1 to M8 are built and merged into `main` (pull requests up to 8). 1,176 offline tests pass; live results are in the root `README.md`.

## Remaining before submission (2026-11-23)
- [ ] Merge the README branch (`docs/krp-readme`) into `main`.
- [ ] Replace the Gemini key at the final commit; confirm no key is in any file or in history.
- [ ] Repo public, or private with view access for `ailaunchpad-dev`.
- [ ] Deck as PDF. It must say which model and date produced the figures, and explain why the golden set tops out at 19 of 20 (EV-020).
- [ ] Video, 10 minutes or less. Turn off standby first (`powercfg /change standby-timeout-ac 0`) and restore it afterwards.
- [ ] Links set to "Anyone with the link can view"; submit through the link in the program email.
