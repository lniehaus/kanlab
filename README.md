# KANLab

An interactive visualization of Kolmogorov-Arnold Networks (KANs), based on Deep Playground,
written in TypeScript using d3.js. We use GitHub issues for tracking new requests and bugs.
Your feedback is highly appreciated!

## ✨ New: Basis-Agnostic Glorot-like Initialization

This playground now implements a theoretically-grounded initialization scheme for KANs that:
- Accounts for B-spline basis function properties
- Preserves variance in both forward and backward passes
- Generalizes Xavier/Glorot, Kaiming/He, and LeCun initialization to KANs
- Uses proper Gaussian sampling for better training dynamics

📚 See [QUICKSTART.md](QUICKSTART.md) for immediate usage or [KAN_INITIALIZATION.md](KAN_INITIALIZATION.md) for details.

**If you'd like to contribute, be sure to review the [contribution guidelines](CONTRIBUTING.md).**

## Development

To run the visualization locally, run:
- `npm i` to install dependencies
- `npm run build` to compile the app and place it in the `dist/` directory
- `npm run serve` to serve from the `dist/` directory and open a page on your browser.

For a fast edit-refresh cycle when developing run `npm run serve-watch`.
This will start an http server and automatically re-compile the TypeScript,
HTML and CSS files whenever they change.

## For owners
To push to production: `git subtree push --prefix dist origin gh-pages`.

This is not an official Google product.
