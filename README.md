# fin
synthetic data in finance papers

## ABIDES: Towards High-Fidelity Multi-Agent Market Simulation (SIGSIM-PADS 2020)

- **Data / Variables:**
  - Nanosecond-timestamped limit order book events (orders, cancels, trades)
  - L1–L2 depth, best bid/ask quotes
  - Agent portfolios and internal states, detailed message logs
  - Pairwise network latency and jitter matrix
  - Optional historical price/volume injection via a “data oracle”

- **Methods:**
  - Interactive **discrete-event** simulation engine
  - Centralized exchange and order book agent with NASDAQ-like ITCH/OUCH messaging
  - Deterministic per-agent random seeds, computation delay, and network latency modeling
  - Library of background agents (e.g., Zero-Intelligence, heuristic/momentum traders)

- **Purpose of Generated Data:**
  - Produce high-frequency **synthetic market microstructure** data (quotes, trades, depth)
  - Enable experiments on co-location effects, market impact, and latency–accuracy trade-offs
  - Test ML-based trading strategies in controlled **counterfactual** scenarios

- **Key Findings / Contributions:**
  - Realistic intraday patterns reproduced by heterogeneous background agents
  - Quantification of **price impact** from large single orders through “what-if” experiments
  - Full visibility into agent identities and intents for explainable analyses

### GANs and Synthetic Financial Data: Calculating VaR (Applied Economics, 2025)

- **Data / Variables:**
  - Daily S&P 500 index values (2012–2022) and FTSE 100 index values (2010–2023)
  - Generated synthetic series using GANs (in index levels, later converted to returns)
  - Descriptive statistics: mean, median, min/max, std. dev., skewness, excess kurtosis, Hurst exponent
  - Higher-order moments (2nd, 3rd, 4th) and cumulants
  - Value-at-Risk (VaR) calculations for 5-day, 5% level

- **Methods:**
  - Wasserstein GAN (WGAN) architecture to generate synthetic index level data
  - Regression analysis, moment and cumulant analysis, Random Forest modeling
  - GARCH(1,1) and ARCH(1) models for volatility
  - Market efficiency tests (lag-1 autocorrelation in returns)
  - Cointegration tests (Engle–Granger) for real vs. synthetic levels
  - VaR estimation using historical simulation

- **Purpose of Generated Data:**
  - Produce realistic synthetic financial time series to:
    - Fill data gaps or create counterfactual scenarios
    - Test trading strategies, market efficiency, and volatility dynamics
    - Estimate VaR and compare real vs. synthetic risk measures

- **Key Findings / Contributions:**
  - Synthetic index **levels** are statistically and cointegration-wise almost indistinguishable from real series
  - Synthetic **returns** diverge in higher-order lag behavior; weak-form market efficiency fails for GAN series
  - GARCH results show lower persistence in volatility for synthetic data
  - Higher explanatory power in regression and Random Forest models for synthetic returns vs. real
  - VaR estimates from synthetic data underestimated actual risk during COVID-19 period
  - Suggestion for hybrid GAN + time series models (e.g., Tail-GAN) to better capture tail risk

- **Notes for Synthetic-Data Workflows:**
  - GANs excel at replicating statistical properties of level series
  - Care needed when transforming to returns—may reveal inefficiencies or unrealistic autocorrelations
  - Useful for scenario simulation, tail-risk analysis, and stress testing with controlled properties

- **Code:**
  - Analysis implemented in Python (PyTorch) and R (quantmod, randomForest, randomForestExplainer)
 
### Generating Realistic Stock Market Order Streams (AAAI-20)

- **Data / Variables:**
  - **Real data:** Millisecond-level limit order streams for GOOG (≈230k orders/day) and a small-cap stock (PN) from OneMarketData
  - **Synthetic data:** 300k orders from an agent-based stock market simulator
  - Each order record (`xᵢ`) includes:  
    - Time interval since last order  
    - Order type (buy, sell, cancel buy, cancel sell)  
    - Limit price, quantity  
    - Best bid/ask prices and quantities at the time of order

- **Methods:**
  - **Stock-GAN:** Conditional Wasserstein GAN (WGAN-GP) with both generator and critic conditioned on:
    - k-length history of past orders (k=20) via LSTM encoding
    - Time-of-day interval (Δt)
  - **Architecture innovations:**
    1. **CDA Network:** Pre-trained neural approximation of the continuous double auction mechanism, embedded in the generator to output updated best bid/ask
    2. **Order Book Context:** Inclusion of order book state in conditioning input to capture hidden market state
  - Loss: Standard WGAN loss + gradient penalty  
  - Training detail: Minibatches constructed with non-overlapping sequences to break dependency

- **Purpose of Generated Data:**
  - Generate high-fidelity synthetic limit order streams for:
    - Market design and regulation studies
    - Backtesting of trading strategies without sensitive real data
    - Benchmarking of market simulation models

- **Evaluation Metrics (Five Statistics):**
  1. Price distribution of orders (by order type)
  2. Quantity distribution (by order type)
  3. Inter-arrival time distribution
  4. Intensity evolution (orders per time chunk)
  5. Best bid/ask evolution over time + spectral density comparison

- **Key Findings / Contributions:**
  - **Performance vs. baselines:** Outperforms RNN-based VAE and DCGAN variants on KS distance for price, quantity, inter-arrival distributions (both synthetic and real data)
  - **Ablation results:** Removing CDA network or order-book input degrades realism (notably in best bid/ask dynamics and spectral properties)
  - **Real data test (GOOG):** Captures distributions and temporal patterns more accurately than baselines; retains high-frequency and low-frequency components closer to real data
  - **Synthetic data test:** Matches stylized facts of the simulator but with greater realism in dynamics
  - Demonstrates that modeling the auction mechanism and market state context is critical for generating realistic order streams

- **Notes for Synthetic-Data Workflows:**
  - Model can be conditioned on exogenous variables (e.g., news events) for scenario generation
  - CDA integration ensures generated orders are consistent with market clearing rules
  - LSTM + order book context enables learning of long-range dependencies in order flow

- **Code:**
  - Data from OneMarketData (licensed), synthetic data from public simulator

### Generation of Synthetic Financial Time Series by Diffusion Models (Neurocomputing, 2024)

- **Data / Variables:**
  - Daily adjusted closing prices of 10 major S&P 500 constituent stocks (2013–2023)
  - Data transformed into log-returns for modeling
  - Statistical properties considered: mean, std. deviation, skewness, kurtosis
  - Distributional shape and autocorrelation structure over different lags

- **Methods:**
  - **Denoising Diffusion Probabilistic Models (DDPM)** for time series generation
  - Data preprocessing: normalization, log-return computation
  - Forward process: Gradual Gaussian noise addition over T timesteps
  - Reverse process: Neural network learns to denoise step-by-step to reconstruct time series
  - Neural architecture: 1D convolutional residual networks for temporal feature extraction
  - Baseline comparisons: GANs, Variational Autoencoders (VAEs)
  - Evaluation metrics:  
    - Kolmogorov–Smirnov (KS) distance for distribution match  
    - Autocorrelation function (ACF) distance  
    - Dynamic Time Warping (DTW) for temporal similarity

- **Purpose of Generated Data:**
  - Produce realistic synthetic stock return series while preserving:
    - Marginal distribution characteristics (mean, variance, skewness, kurtosis)
    - Temporal dependencies (autocorrelation, volatility clustering)
  - Enable data augmentation, backtesting, and stress-testing scenarios without using sensitive data

- **Key Findings / Contributions:**
  - Diffusion models outperform GANs and VAEs in matching both distributional and temporal statistics
  - Generated returns retain realistic autocorrelation decay patterns
  - Lower KS distances and DTW scores compared to baselines
  - Better at capturing volatility clustering than GANs/VAEs
  - Can model heavy tails more effectively due to iterative denoising process
  - Demonstrates scalability to multiple correlated assets with joint training

- **Notes for Synthetic-Data Workflows:**
  - Diffusion models are particularly robust for financial time series with noise and non-linear dependencies
  - Training stability is higher compared to GANs, avoiding mode collapse
  - Potential to integrate with conditional signals (e.g., macroeconomic indicators) for scenario generation

- **Code**
  - Implemented in PyTorch
 
### Limit Order Book Dynamics and Order Size Modelling Using Compound Hawkes Process (Quantitative Finance, 2020)

- **Data / Variables:**
  - Level-1 limit order book (LOB) data for 5 liquid stocks from NASDAQ Nordic over 3 months
  - Event types: market buy, market sell, limit buy, limit sell, cancel buy, cancel sell
  - Event timestamps with millisecond resolution
  - Order sizes and associated price changes
  - Mid-price series derived from best bid/ask quotes

- **Methods:**
  - **Compound Hawkes Process (CHP):**  
    - Hawkes process models the arrival times of LOB events with self-excitation and mutual-excitation kernels  
    - Compound component models order sizes as marks attached to events
  - Kernel function: Exponential decay to capture short-term clustering of events
  - Parametric models for order size distribution (Gamma, Lognormal)
  - Calibration using maximum likelihood estimation (MLE)
  - Goodness-of-fit evaluation with residual analysis and QQ-plots

- **Purpose of Generated Data:**
  - Produce synthetic LOB event streams that replicate:
    - Realistic event arrival clustering (self- and cross-excitation effects)
    - Empirical order size distributions
    - Impact on short-term price dynamics and volatility
  - Enable microstructure simulation and algorithmic trading backtesting with realistic event timing and size patterns

- **Key Findings / Contributions:**
  - CHP effectively reproduces empirical clustering patterns in event arrivals
  - Captures heavy-tailed distribution of order sizes better than simple Poisson or standard Hawkes models
  - Simulation of synthetic event streams yields mid-price dynamics similar to real LOB data
  - Mutual-excitation between buy/sell orders plays a significant role in volatility persistence
  - Model provides a flexible tool for joint timing–size modeling of LOB events

- **Notes for Synthetic-Data Workflows:**
  - CHP is well-suited for event-driven simulation where both timing and size matter
  - Can be integrated into agent-based market models or used standalone for order flow generation
  - Parameter interpretability enables linking microstructure features to trader behavior

### Limit Order Book Simulation with Generative Adversarial Networks (2021)

- **Data / Variables:**
  - Real high-frequency LOB data for 5 U.S. equities from NASDAQ ITCH dataset
  - Variables include:
    - Order type (limit, market, cancel)
    - Price level, size, side (bid/ask)
    - Event timestamp (nanosecond resolution)
    - Depth snapshots capturing top-10 bid/ask levels

- **Methods:**
  - **GAN-based simulation framework** for generating synthetic LOB states and dynamics
  - Generator: Convolutional + recurrent layers to capture both spatial (price levels) and temporal (order arrival) patterns
  - Discriminator: Convolutional architecture distinguishing real vs. generated LOB states
  - Conditional generation: Model conditioned on previous LOB states to maintain temporal consistency
  - Comparison baselines: Vector AutoRegressive (VAR) models, standard GANs without conditioning
  - Evaluation metrics:
    - Distributional similarity for price changes, volume, spread
    - Auto-correlation of returns and order flows
    - Stylized facts reproduction (volatility clustering, heavy tails)

- **Purpose of Generated Data:**
  - Create realistic synthetic LOB sequences for:
    - Stress-testing trading algorithms
    - Microstructure research without using proprietary data
    - Market impact and liquidity risk studies

- **Key Findings / Contributions:**
  - Conditional GAN better captures temporal dependencies in order flow than vanilla GAN
  - Successfully reproduces stylized facts observed in real LOB data:
    - Heavy-tailed return distributions
    - Volatility clustering
    - Long-memory in order sign autocorrelation
  - Outperforms VAR and unconditional GAN in distributional metrics and time-series properties
  - Generated data exhibits realistic spread and depth dynamics

- **Notes for Synthetic-Data Workflows:**
  - GAN-based LOB simulators can replace or complement agent-based models
  - Conditioning on past states is essential to preserve market microstructure features
  - Model could be extended with exogenous conditioning variables (e.g., news or macro events)

- **Code / License:**
  - Implemented in TensorFlow
  - Dataset from NASDAQ ITCH feed (licensed, not public)

### Neural Stochastic Agent-Based Limit Order Book Simulation with Neural Point Processes (2023)

- **Data / Variables:**
  - High-frequency LOB event data from NASDAQ TotalView-ITCH feed
  - Event attributes:
    - Event type (limit buy, limit sell, cancel, market buy, market sell)
    - Price level and order size
    - Side (bid/ask)
    - Nanosecond timestamps
  - Derived variables: mid-price, spread, depth imbalance

- **Methods:**
  - **Neural stochastic agent-based simulation** combining:
    - Multi-agent environment with heterogeneous trading strategies
    - **Neural Point Processes (NPP)** to model event arrival times conditioned on historical order flow
    - Graph neural networks (GNNs) to represent LOB state as a spatial–temporal graph
  - Each agent's decision-making modeled as a stochastic policy conditioned on the LOB state embedding
  - Event generation pipeline:
    1. NPP samples next event time
    2. Agent policy selects action type and parameters
    3. LOB state updated and fed back into agent models
  - Baselines: Standard Hawkes process, ABIDES without neural enhancements
  - Evaluation metrics:
    - Stylized fact reproduction (return distribution, autocorrelation, volatility clustering)
    - Order flow intensity prediction accuracy
    - Spread and depth dynamics realism

- **Purpose of Generated Data:**
  - Produce synthetic LOB event streams capturing both:
    - Realistic temporal clustering of events
    - Strategic interactions between heterogeneous agents
  - Enable high-fidelity market microstructure simulations with adaptive agents

- **Key Findings / Contributions:**
  - Neural-enhanced ABM outperforms traditional Hawkes and rule-based ABIDES in reproducing microstructure stylized facts
  - GNN-based LOB embeddings improve predictive accuracy of next-event characteristics
  - NPP effectively models varying event intensities across trading day
  - Demonstrates the feasibility of integrating deep learning with ABM for market simulation

- **Notes for Synthetic-Data Workflows:**
  - Combines the interpretability of agent-based models with the flexibility of neural temporal models
  - Can be extended with reinforcement learning for adaptive agent training
  - Requires large-scale, high-quality event data for calibration

- **Code / License:**
  - No public code release
  - Dataset from NASDAQ ITCH feed (licensed)
 
### Quant GANs: Deep Generation of Financial Time Series (2020)

- **Data / Variables:**
  - Daily OHLCV data for S&P 500 constituents
  - Variables: Open, High, Low, Close prices, Volume
  - Derived metrics: log returns, volatility, autocorrelation structures

- **Methods:**
  - **Quant GAN** framework:
    - Generator: Temporal Convolutional Network (TCN) producing synthetic multivariate time series
    - Discriminator: CNN-based architecture distinguishing real vs. synthetic sequences
    - Training objective incorporates **Wasserstein GAN with Gradient Penalty (WGAN-GP)**
    - **Quantitative constraints** included in the loss function to enforce stylized facts (e.g., heavy tails, volatility clustering)
  - Data preprocessing:
    - Normalization by rolling window statistics
    - Return-based transformation for stationarity
  - Evaluation:
    - Statistical similarity tests (Kolmogorov–Smirnov, autocorrelation comparison)
    - Financial metrics preservation (Sharpe ratio, maximum drawdown)
    - Out-of-sample backtesting of trading strategies on generated data

- **Purpose of Generated Data:**
  - Create realistic synthetic price series preserving essential financial properties
  - Augment training data for ML-based trading models
  - Enable scenario analysis and risk modeling without exposing proprietary data

- **Key Findings / Contributions:**
  - Quant GAN produces synthetic series statistically indistinguishable from real data in key distributional metrics
  - Stylized facts (fat tails, volatility clustering) reproduced more accurately than with standard GAN or TCN baselines
  - Demonstrated usefulness for training predictive models with improved generalization

- **Notes for Synthetic-Data Workflows:**
  - Incorporating domain-specific constraints into GAN training improves realism
  - Suitable for low-frequency financial time series; adaptation needed for high-frequency LOB data
  - Can be combined with copula models for cross-asset correlation preservation

- **Code / License:**
  - Dataset derived from publicly available S&P 500 historical prices
 
### Synthetic Data Applications in Finance (2022)

- **Data / Variables:**
  - Broad review of data types in finance suitable for synthetic generation:
    - Market microstructure data (limit order books, trades, quotes)
    - Price and return time series (intraday, daily, multi-asset)
    - Risk factor series (interest rates, volatilities)
    - Transaction and customer-level data for retail banking
  - Variables depend on application domain:
    - For trading: price, volume, spread, depth
    - For risk: VaR, Expected Shortfall, factor loadings
    - For retail finance: demographics, transaction categories, balances

- **Methods:**
  - Overview of synthetic data generation techniques:
    - **Generative models**: GANs, VAEs, diffusion models
    - **Statistical models**: GARCH, copulas, Hawkes processes
    - **Agent-based simulations** for market microstructure
  - Emphasis on **privacy-preserving** synthetic data generation (differential privacy, k-anonymity)
  - Model evaluation:
    - Fidelity metrics (distributional similarity, autocorrelation)
    - Utility metrics (downstream ML task performance)
    - Privacy risk metrics (re-identification probability)

- **Purpose of Generated Data:**
  - Mitigate data scarcity and access restrictions due to confidentiality
  - Support research and development of:
    - Algorithmic trading models
    - Risk management systems
    - Fraud detection algorithms
  - Enable scenario analysis and stress testing without exposing sensitive data

- **Key Findings / Contributions:**
  - Synthetic data is most impactful when:
    - Real data access is legally or operationally restricted
    - The application requires rare-event modeling (e.g., crisis scenarios)
  - High-fidelity synthetic data can match real data performance in ML tasks
  - Privacy–utility trade-off remains a central challenge
  - Regulatory acceptance is growing but requires transparency in generation methods

- **Notes for Synthetic-Data Workflows:**
  - Selection of generation method should match data type and application
  - Validation must include both statistical similarity and downstream performance
  - Privacy-preserving guarantees increasingly important for adoption

- **Code:**
  - Review paper
  - Examples reference public datasets (LOBSTER, Yahoo Finance, ECB statistics)

### TRADES: Generating Realistic Market Simulations with Diffusion Models (2024)

- **Data / Variables:**
  - Limit Order Book (LOB) event data from cryptocurrency markets (BTC/ETH pairs)
  - Variables:
    - Event type (limit buy/sell, cancel, market buy/sell)
    - Price, volume
    - Timestamps
    - Order book depth snapshots
  - Derived features: mid-price returns, spread, depth imbalance

- **Methods:**
  - **TRADES framework**:
    - Uses **Diffusion Probabilistic Models (DPMs)** to generate synthetic sequences of LOB states
    - Models joint evolution of order arrivals, cancellations, and executions
    - Incorporates temporal encoding to preserve high-frequency event dependencies
    - Conditioning on macro market states (e.g., volatility regime) for scenario-specific generation
  - Data preprocessing:
    - Event aggregation into fixed time intervals
    - Normalization of price and volume features
  - Evaluation metrics:
    - Stylized fact reproduction (heavy tails, volatility clustering, long-memory in order flow)
    - Distributional similarity tests
    - Downstream trading strategy performance

- **Purpose of Generated Data:**
  - Enable realistic simulation of LOB dynamics for:
    - Backtesting algorithmic strategies
    - Risk management under extreme conditions
    - Training reinforcement learning agents

- **Key Findings / Contributions:**
  - Diffusion-based generation outperforms GAN and Hawkes baselines in:
    - Capturing complex temporal correlations
    - Reproducing higher-order LOB statistics
  - Conditioning improves controllability for scenario simulation
  - Generated data supports RL training with comparable performance to real data

- **Notes for Synthetic-Data Workflows:**
  - Diffusion models are robust to mode collapse and can better handle multi-modal order flow patterns
  - Conditioning mechanism allows stress-testing under user-defined regimes
  - Requires careful tuning of temporal encoding for high-frequency markets

- **Code / License:**
  - Crypto LOB dataset proprietary but similar data available from public crypto exchanges

### TSGBench: Time Series Generation Benchmark (2023)

- **Data / Variables:**
  - Multiple real-world time series datasets across domains (including finance, energy, healthcare)
  - Financial datasets include:
    - Stock price OHLC data
    - Trading volumes
    - Derived returns and volatility measures
  - Variables differ per dataset but generally include timestamped numerical sequences

- **Methods:**
  - **TSGBench framework**:
    - Standardized benchmark for evaluating **time series generation models**
    - Includes classical statistical models (ARIMA, VAR), deep generative models (GANs, VAEs, Diffusion Models), and hybrid approaches
    - Unified evaluation protocol with consistent preprocessing and metrics
  - Metrics used:
    - **Fidelity**: Distributional similarity, autocorrelation, spectral density comparison
    - **Diversity**: Coverage of possible patterns, mode collapse detection
    - **Utility**: Performance on downstream predictive tasks (e.g., forecasting)
  - Benchmark tasks:
    - Unconditional generation
    - Conditional generation given partial sequences

- **Purpose of Generated Data:**
  - Provide a fair, reproducible way to compare generative models for time series
  - Highlight trade-offs between fidelity, diversity, and downstream utility
  - Guide researchers in selecting models for specific applications (finance, healthcare, etc.)

- **Key Findings / Contributions:**
  - No single model dominates across all domains and metrics
  - Diffusion models generally excel in fidelity, GAN variants in diversity
  - Domain-specific tuning is essential for optimal results
  - TSGBench helps identify model weaknesses (e.g., temporal dependency preservation)

- **Notes for Synthetic-Data Workflows:**
  - Benchmarking generative models on multiple datasets prevents overfitting to a single domain
  - Financial time series benefit from domain-specific loss terms and constraints
  - Balanced evaluation (fidelity + utility) avoids misleading conclusions

- **Code**
  - Publicly available benchmark code and datasets on GitHub
  - Modular design allows easy addition of new datasets and models

### Time-series Generative Adversarial Networks (TimeGAN) (2019)

- **Data / Variables:**
  - General time series framework, tested on synthetic and real datasets
  - Financial data example:
    - Stock prices and returns
    - Multiple correlated asset time series
  - Variables: timestamps, multi-dimensional sequences of numerical values

- **Methods:**
  - **TimeGAN architecture**:
    - Combines **GAN** framework with **recurrent neural networks (RNNs)** for temporal modeling
    - Uses a **supervised loss** to preserve temporal dynamics and correlations
    - Architecture components:
      - Embedding network: maps real sequences to latent space
      - Recovery network: reconstructs data from latent space
      - Generator: produces synthetic latent sequences
      - Discriminator: distinguishes real vs synthetic sequences
      - Supervisor: enforces temporal dependencies in generated data
    - Joint optimization of supervised, adversarial, and reconstruction losses
  - Evaluation metrics:
    - Fidelity: distributional similarity (PCA, t-SNE visualization, statistical tests)
    - Predictive score: performance of forecasting models trained on synthetic data
    - Discriminative score: classifier’s accuracy in distinguishing real vs synthetic

- **Purpose of Generated Data:**
  - Preserve both distributional characteristics and temporal dynamics of multivariate time series
  - Provide realistic synthetic data for:
    - Risk modeling
    - Algorithmic trading
    - Stress testing
    - ML model training when real data is scarce

- **Key Findings / Contributions:**
  - TimeGAN outperforms standard GANs and VAEs in preserving temporal dependencies
  - Supervisory loss is crucial for long-term dependency capture
  - Generated data supports training of downstream models with high utility

- **Notes for Synthetic-Data Workflows:**
  - Particularly effective for high-dimensional correlated sequences
  - Requires careful hyperparameter tuning for balance between adversarial and supervised loss
  - Can be extended with domain-specific features or constraints
