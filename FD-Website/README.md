# Fraud Detection (FD) — Single-page website

## Files
- `index.html` — all website sections and navigation
- `style.css` — responsive dark dashboard styling
- `script.js` — navigation, inference graphs and demo form

## Run locally
Open `index.html` in a modern browser. No build tools are required.

## Publish with GitHub Pages
1. Copy these three files into the root of your `College-project` repository.
2. Commit and push the files to the `main` branch.
3. In GitHub, open **Settings → Pages**.
4. Under **Build and deployment**, select **Deploy from a branch**.
5. Choose branch `main` and folder `/ (root)`, then save.
6. Wait for the Pages deployment and open the published URL.

## Important project notes
- The benchmark table contains the selected five model rows from the supplied project README.
- The inference table and three graphs use the five uploaded inference result files.
- The inference result files did not consistently state their dataset, so the website does not assign dataset names to those values.
- The transaction form is a front-end demonstration only. It does not run a trained Python model. A Python backend/API must be deployed and connected for genuine live fraud predictions.
- Large model weights (for example, a 134 MB `.pth` file) exceed GitHub's normal 100 MB per-file limit. Do not commit them through ordinary Git; use Git LFS or host the model separately if needed.
