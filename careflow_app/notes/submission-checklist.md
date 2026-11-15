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
- [ ] Code does **not rely on local file paths**.
- [ ] Presentation is a PDF; video is <= 10 minutes.
- [ ] (Pair only) partner's full name and email ready.

## Project-specific notes
- Do not commit `.env` or `data/` (toolkit ground rule). If the repo is public, check no API key is in any file or history.
- Current `careflow_app` reads the data and key from `../capstone-data-toolkit/` via paths relative to the file (not absolute), but it still expects the toolkit's `.env` and generated `data/` to exist. Before submitting, document in the README how a reviewer regenerates the data (`python generate.py --domain careflow`) or ship the data.
- Dates differ: the project brief says final presentations are 15-16 August, but this email's deadline is 25 Nov 2026. Treat the email as current; confirm with the program team if unsure.
- Milestone status: M1 in progress (code built, 5-record smoke test done, full 200-record eval pending). M2-M8 not started.
