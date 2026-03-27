# Neural Ordinary Differential Equations
### *Continuous-Depth Models When Infinite Layers Cost O(1) Memory*


---

## What This Is

A tutorial on **Neural ODEs** (Chen et al., 2018) models where a neural network parameterises the *derivative* of the hidden state, and a black-box ODE solver integrates it forward in continuous time.

Covers: the ResNet → Neural ODE connection, numerical solvers (Euler/RK4/dopri5), the adjoint method for O(1) memory training, spiral classification as a continuous flow, and irregular time series as a unique capability.

**Key insight:** A Neural ODE is the continuous limit of an infinitely deep ResNet depth becomes adaptive, memory becomes O(1), and the model can handle data at any time stamps.

---

## Repository Contents

| File | Description |
|---|---|
| `node_tutorial.docx` | Full tutorial document (Word format, <2000 words) |
| `node_tutorial.ipynb` | Jupyter notebook all code, figures, model implementation |
| `README.md` | This file |
| `LICENSE` | MIT licence |

---

## How to Run

```bash
pip install numpy matplotlib scipy torch torchdiffeq nbformat
jupyter notebook node_tutorial.ipynb
```

`torchdiffeq` is optional if not installed, the ODE function cell defines the interface but skips the actual `odeint` call. All figures still run without it.

**Google Colab:**
```python
!pip install torchdiffeq
```
Then upload and run `node_tutorial.ipynb`.

---

## Figures

| Figure | Description |
|---|---|
| `node_fig1_resnet_vs_ode.png` | ResNet discrete layer diagram + Neural ODE continuous trajectory with vector field |
| `node_fig2_solvers_adjoint.png` | Euler vs RK4 accuracy + adjoint method O(1) memory diagram |
| `node_fig3_spiral.png` | Three-panel spiral classification: before, vector field, after training |
| `node_fig4_timeseries.png` | Irregular time series interpolation + accuracy vs NFE trade-off |

---

## Accessibility

- Colourblind-safe palette: burgundy (#7B2D3E), rust (#C0542A), sage (#4A7C59)
- Spiral classification uses **both colour and shape markers** (circles vs squares)
- Line plots use **distinct solid/dashed styles**
- All images carry full **descriptive alt text**
- Section headings follow a **1→4 hierarchy** for screen-reader navigation
- Code blocks use Courier New at consistent readable size

---

## References

1. Chen, R.T.Q. et al. (2018) *Neural ordinary differential equations* https://arxiv.org/abs/1806.07366
2. Rubanova, Y. et al. (2019) *Latent ODEs for irregularly-sampled time series* https://arxiv.org/abs/1907.03907
3. Grathwohl, W. et al. (2019) *FFJORD* — https://arxiv.org/abs/1810.01367
4. He, K. et al. (2016) *Deep residual learning for image recognition* https://arxiv.org/abs/1512.03385
5. Pontryagin, L.S. et al. (1962) *The Mathematical Theory of Optimal Processes*. Wiley-Interscience.

---

