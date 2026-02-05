# 구현 가이드: Sparse-Aware Spatio-Temporal Demand Forecasting Model

[cite_start]이 문서는 희소한 수요 데이터(Sparse Demand Data)를 예측하기 위해 설계된 시공간 딥러닝 모델의 구현 명세서입니다[cite: 1]. [cite_start]이 모델은 **Snapshot-Based Global Attention**과 **OD(Origin-Destination) 데이터 기반의 편향(Bias)**을 활용하는 것이 핵심입니다[cite: 1, 77].

---

## 1. 하이퍼파라미터 및 상수 (Constants)
구현 전 설정해야 할 주요 차원(Dimension) 값입니다.

* [cite_start]**Static Feature Dim ($d_s$):** 32 [cite: 29]
* [cite_start]**Temporal Feature Dim ($d_t$):** 16 [cite: 35]
* [cite_start]**Dynamic Feature Dim ($d_d$):** 32 [cite: 62]
* [cite_start]**Combined Embedding Dim ($e_{r,t}$):** 64 [cite: 64]
* [cite_start]**Attention Heads ($H$):** 4 [cite: 74]
* [cite_start]**Head Dimension ($d_h$):** 16 [cite: 74]
* [cite_start]**Lag Window ($l$):** 예: 4 [cite: 46]
* [cite_start]**Sequence Length ($T$):** 예: 24 (시간 단위) [cite: 92]

---

## 2. 모듈별 상세 구현 (Architecture Modules)

[cite_start]모델은 크게 **입력 인코더**, **시공간 처리(RNN + Attention)**, **출력 헤드**로 구성됩니다[cite: 2, 65, 100].

### 2.1. 정적 지역 인코더 (Static Region Encoder)
[cite_start]각 지역(Region) $r$의 고정된 특성을 처리합니다[cite: 3, 4].

* **입력 데이터 ($x_r$):**
    1.  [cite_start]**Land-use:** $C_{land}$ 차원 벡터 (정규화됨)[cite: 5, 7].
    2.  [cite_start]**POI (Points of Interest):** 단순 개수가 아닌 **존재 여부($z$)**와 **크기($s$)**로 분해하여 처리합니다[cite: 13, 15].
        * [cite_start]$z_{r,c} = \mathbb{I}(x_{r,c}^{POI} > 0)$ [cite: 16]
        * [cite_start]$s_{r,c} = \log(1 + x_{r,c}^{POI})$ [cite: 16]
        * [cite_start]각각에 대해 학습 가능한 가중치 $w^{(z)}, w^{(s)}$를 적용합니다 (`Softplus`로 양수 보장)[cite: 17, 18, 19].
        * [cite_start]최종 POI 특징: $\tilde{x}_{r,c}^{POI} = w_{c}^{(z)}z_{r,c} + w_{c}^{(s)}s_{r,c}$[cite: 21].
    3.  [cite_start]**Geographic:** 위도, 경도, $\log(1+\text{area})$[cite: 23, 24].
* [cite_start]**처리:** 위 특징들을 `concat`한 후 MLP($f_{stat}$)를 통과시켜 $u_r^{stat}$ 생성 ($d_s=32$)[cite: 27, 29].

### 2.2. 시간적 맥락 인코딩 (Temporal Context Encoding)
[cite_start]특정 시점 $t$의 시간 정보를 임베딩합니다[cite: 28].

* **구성 요소:**
    * [cite_start]**요일 (Day-of-Week):** Lookup Table ($7 \times d_t$)[cite: 31, 33].
    * [cite_start]**시간 (Hour-of-Day):** Lookup Table ($24 \times d_t$)[cite: 37, 38].
    * [cite_start]**공휴일 (Holiday):** 공휴일일 경우 더해지는 학습 가능한 벡터 $e^{hol}$[cite: 40, 41].
* [cite_start]**출력:** $u_t^{time} = u_t^{dow} + u_t^{hod} + \mathbb{I}(holiday(t)) \cdot e^{hol}$[cite: 43].

### 2.3. 동적 수요 인코더 (Dynamic Demand Encoder)
[cite_start]과거 수요 패턴을 인코딩합니다[cite: 44].

* **입력 데이터:**
    * [cite_start]**Lag Sequence:** 길이 $l$의 과거 수요 $y_{r,t}^{(l)}$[cite: 46, 47].
    * [cite_start]**Mask:** 패딩 여부 $m_{r,t,k}$[cite: 50].
    * [cite_start]**Sparsity Descriptor:** 0이 아닌 수요의 개수 $c_{r,t}$[cite: 53].
    * [cite_start]**Global Recency ($\Delta t_{r,t}^{last}$):** **(중요)** 현재 윈도우와 무관하게, 마지막으로 수요가 있었던 시점으로부터 경과한 시간[cite: 54, 55]. [cite_start]너무 큰 값은 $\Delta_{max}$로 클리핑[cite: 59].
* [cite_start]**처리:** 위 특징들을 `concat` 후 MLP($f_{dyn}$)를 통과시켜 $u_{r,t}^{dyn}$ 생성 ($d_d=32$)[cite: 61, 62].

---

## 3. 시공간 모델링 (Main Spatio-Temporal Loop)

각 타임스텝 $t$마다 다음 과정을 순차적으로 수행합니다.

#### Step 1: 초기 결합 (Initial Embedding)
$$e_{r,t} = \phi(W_e [u_r^{stat} || u_{r,t}^{dyn} || u_t^{time}] + b_e)$$
* [cite_start]결과 차원: 64[cite: 64].

#### Step 2: 시간적 상태 업데이트 (GRU & Gated Fusion)
* [cite_start]**GRU:** 이전 상태 $h_{r, t-1}$과 현재 입력 $e_{r,t}$를 사용해 업데이트[cite: 67].
* [cite_start]**Gated Fusion:** 순간적인 정보($e$)와 역사적 정보($h$)를 융합[cite: 68].
    * [cite_start]Gate $g_{r,t} = \sigma(W_g [e_{r,t} || h_{r,t}] + b_g)$ [cite: 69]
    * [cite_start]Fused State $s_{r,t} = g_{r,t} \odot h_{r,t} + (1-g_{r,t}) \odot e_{r,t}$[cite: 70].

#### Step 3: Snapshot-Based Global Attention (Spatial Modeling)
지역 간 상호작용을 계산합니다. [cite_start]**OD 데이터 기반 Bias**가 핵심입니다[cite: 71, 77].

1.  [cite_start]**Query, Key, Value:** $s_{r,t}$로부터 생성 ($H=4, d_h=16$)[cite: 74, 75].
2.  **Soft OD Bias ($g_{OD}$):**
    * [cite_start]OD 데이터 로그 변환: $x_{rj,t} = \log(1 + OD_{r \to j, t})$[cite: 78].
    * [cite_start]정규화 (시간 $t$별 평균/표준편차): $\tilde{OD}_{rj,t} = \frac{x_{rj,t} - \mu_{\overline{OD}}(t)}{\sigma_{OD}(t) + \epsilon}$[cite: 82].
    * [cite_start]가중치 적용: $g_{OD}(r,j,t) = \text{Softplus}(\theta_{OD}) \cdot \tilde{OD}_{rj,t}$[cite: 83, 84].
3.  **Attention Score:**
    $$s_{rj,t}^{(h)} = \frac{(q_{r,t}^{(h)})^{\top}k_{j,t}^{(h)}}{\sqrt{d_h}} + g_{OD}(r,j,t)$$
    [cite_start][cite: 86].
4.  [cite_start]**Aggregation & Output:** Softmax 적용 후 Value와 결합 ($\tilde{s}_{r,t}$)[cite: 88].
5.  [cite_start]**Residual Update:** $\overline{s}_{r,t} = \text{LayerNorm}(s_{r,t} + \tilde{s}_{r,t})$[cite: 90].

---

## 4. 시간적 집계 (Temporal Aggregation)

[cite_start]출력을 내기 전, 윈도우 $T$ 내의 모든 히든 스테이트를 요약합니다[cite: 91].

* [cite_start]각 시점 $\tau \in \{t-T+1, ..., t\}$에 대해 Attention Weight $\beta_{r,\tau}$를 계산합니다[cite: 98].
* [cite_start]가중 합을 통해 최종 컨텍스트 벡터 $\overline{h}_{\tau}$를 생성합니다[cite: 98].

---

## 5. Sparse-Aware Output Heads

[cite_start]희소성을 해결하기 위해 **발생 확률**과 **발생 시 크기**를 분리하여 예측합니다[cite: 100].

1.  [cite_start]**Event Probability ($p$):** `Sigmoid` 활성화 함수 사용[cite: 103].
    * $p_{r,t+1} = \sigma(w_p^{\top}\overline{h}_{r} + b_p)$
2.  [cite_start]**Conditional Magnitude ($\hat{y}^+$):** `Softplus` 활성화 함수 사용[cite: 104].
    * $\hat{y}_{r,t+1}^{+} = \text{Softplus}(w_{+}^{\top}\overline{h}_{r} + b_{+})$
3.  [cite_start]**Final Prediction:** $\hat{y}_{r,t+1} = p_{r,t+1} \times \hat{y}_{r,t+1}^+$[cite: 105].

---

## 6. 손실 함수 (Loss Function)

$$\mathcal{L}_{total} = \mathcal{L}_{evt} + \mathcal{L}_{mag} + \eta \lambda_{OD}^2$$

1.  [cite_start]**Event Loss ($\mathcal{L}_{evt}$):** Binary Cross Entropy (수요 발생 여부 분류)[cite: 108, 109].
2.  [cite_start]**Magnitude Loss ($\mathcal{L}_{mag}$):** 실제 수요가 있는 경우($\delta_{r,t+1}=1$)에만 계산[cite: 111, 113].
    * 절대 오차와 상대 오차의 결합:
        $$\mathcal{L}_{mag} = \sum_{r,t} \delta_{r,t+1} \left( \frac{|\hat{y}_{r,t+1}^{+} - y_{r,t+1}|}{1 + y_{r,t+1}} + \lambda |\hat{y}_{r,t+1}^{+} - y_{r,t+1}| \right)$$
3.  [cite_start]**Regularization:** OD 가중치($\lambda_{OD}$)에 대한 $L_2$ 규제 ($\eta=10^{-3}$)[cite: 115].