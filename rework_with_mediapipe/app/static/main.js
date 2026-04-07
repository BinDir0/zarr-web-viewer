const appNode = document.getElementById("app");

if (appNode) {
  const clipId = Number(appNode.dataset.clipId);
  const frameImage = document.getElementById("frame-image");
  const overlayCanvas = document.getElementById("overlay-canvas");
  const frameSlider = document.getElementById("frame-slider");
  const frameLabel = document.getElementById("frame-label");
  const playToggle = document.getElementById("play-toggle");
  const submitButton = document.getElementById("submit-review");
  const submitMessage = document.getElementById("submit-message");
  const leftChoices = document.getElementById("left-choices");
  const rightChoices = document.getElementById("right-choices");
  const candidateCards = document.getElementById("candidate-cards");
  const titleNode = document.getElementById("title");
  const mergeCard = document.getElementById("merge-card");
  const mergeQuestionsNode = document.getElementById("merge-questions");

  let bundle = null;
  let currentFrame = 0;
  let playing = false;
  let timerId = null;
  let selectedLeft = null;
  let selectedRight = null;
  const mergeAnswers = new Map();

  function buildChoices(container, side, options, onSelect) {
    container.innerHTML = "";
    const group = document.createElement("div");
    group.className = "choice-group";
    options.forEach((option) => {
      const button = document.createElement("button");
      button.className = "choice-btn";
      button.textContent = option.label;
      button.onclick = () => onSelect(option.value);
      button.dataset.value = option.value;
      group.appendChild(button);
    });
    container.appendChild(group);
  }

  function updateChoiceStyles() {
    document.querySelectorAll("#left-choices .choice-btn").forEach((button) => {
      button.classList.toggle("is-selected", button.dataset.value === selectedLeft);
    });
    document.querySelectorAll("#right-choices .choice-btn").forEach((button) => {
      button.classList.toggle("is-selected", button.dataset.value === selectedRight);
    });
  }

  function drawFrame(frameIndex) {
    if (!bundle) return;
    const frame = bundle.frames[frameIndex];
    if (!frame) return;
    frameImage.src = `/assets/${frame.relpath}`;
    frameLabel.textContent = `${frameIndex + 1} / ${bundle.frames.length}`;
    frameSlider.value = frameIndex;
    currentFrame = frameIndex;
    frameImage.onload = () => {
      overlayCanvas.width = frameImage.clientWidth;
      overlayCanvas.height = frameImage.clientHeight;
      const ctx = overlayCanvas.getContext("2d");
      const naturalWidth = frameImage.naturalWidth || 1;
      const naturalHeight = frameImage.naturalHeight || 1;
      const sx = overlayCanvas.width / naturalWidth;
      const sy = overlayCanvas.height / naturalHeight;
      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      bundle.chains.forEach((chain, chainIdx) => {
        const proposal = chain.proposals.find((item) => item.frame_idx === frame.frame_idx);
        if (!proposal) return;
        const color = chainIdx === Number(selectedLeft?.replace("chain:", "")) ? "#10b981" :
          chainIdx === Number(selectedRight?.replace("chain:", "")) ? "#ef4444" :
          "#3b82f6";
        const [x1, y1, x2, y2] = proposal.bbox_xyxy;
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
        ctx.fillStyle = color;
        ctx.font = "14px sans-serif";
        ctx.fillText(`C${chainIdx}`, x1 * sx + 4, y1 * sy + 16);
      });
    };
  }

  function renderCards() {
    candidateCards.innerHTML = "";
    bundle.card_view.forEach((card) => {
      const node = document.createElement("article");
      node.className = "candidate-card";
      node.innerHTML = `
        <img src="/assets/${card.preview_relpath}" alt="candidate preview">
        <h3>候选 ${card.chain_index}</h3>
        <div class="meta-note">frames ${card.start_frame}-${card.end_frame} · ${card.num_frames} 帧</div>
        <div class="meta-note">${card.role_hint} · score ${card.score.toFixed(2)}</div>
      `;
      candidateCards.appendChild(node);
    });
  }

  function renderMergeQuestions() {
    if (!bundle.merge_questions || bundle.merge_questions.length === 0) {
      mergeCard.classList.add("hidden");
      return;
    }
    mergeCard.classList.remove("hidden");
    mergeQuestionsNode.innerHTML = "";
    bundle.merge_questions.forEach((question) => {
      const wrapper = document.createElement("div");
      wrapper.className = "meta-note";
      const yes = document.createElement("button");
      const no = document.createElement("button");
      yes.className = "choice-btn";
      no.className = "choice-btn is-muted";
      yes.textContent = "是";
      no.textContent = "否";
      yes.onclick = () => {
        mergeAnswers.set(question.question_id, true);
        yes.classList.add("is-selected");
        no.classList.remove("is-selected");
      };
      no.onclick = () => {
        mergeAnswers.set(question.question_id, false);
        no.classList.add("is-selected");
        yes.classList.remove("is-selected");
      };
      wrapper.innerHTML = `<div>${question.prompt}</div>`;
      wrapper.appendChild(yes);
      wrapper.appendChild(no);
      mergeQuestionsNode.appendChild(wrapper);
    });
  }

  async function submitReview() {
    if (!selectedLeft || !selectedRight) {
      submitMessage.textContent = "请先选择 left / right。";
      return;
    }
    if (selectedLeft === selectedRight && !["none", "unsure"].includes(selectedLeft)) {
      submitMessage.textContent = "左右手不能选同一候选。";
      return;
    }
    const payload = {
      left_choice: selectedLeft,
      right_choice: selectedRight,
      merge_answers: Array.from(mergeAnswers.entries()).map(([question_id, answer]) => ({question_id, answer})),
      review_confidence: document.getElementById("review-confidence").value,
    };
    const response = await fetch(`/api/clips/${clipId}/submit-review`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok || !data.success) {
      submitMessage.textContent = data.message || "提交失败";
      return;
    }
    submitMessage.textContent = "提交成功，后台将异步拟合 MANO。";
  }

  function togglePlayback() {
    playing = !playing;
    playToggle.textContent = playing ? "暂停" : "播放";
    if (!playing) {
      if (timerId) window.clearInterval(timerId);
      timerId = null;
      return;
    }
    timerId = window.setInterval(() => {
      const nextFrame = (currentFrame + 1) % bundle.frames.length;
      drawFrame(nextFrame);
    }, 120);
  }

  async function boot() {
    const response = await fetch(`/api/clips/${clipId}`);
    const data = await response.json();
    bundle = data.bundle;
    titleNode.textContent = `${bundle.episode_name} · frames ${bundle.clip_start}-${bundle.clip_end}`;
    const options = bundle.card_view.map((item) => ({
      label: `候选 ${item.chain_index}`,
      value: `chain:${item.chain_index}`,
    }));
    options.push({label: "没出现", value: "none"});
    options.push({label: "不确定", value: "unsure"});
    buildChoices(leftChoices, "left", options, (value) => { selectedLeft = value; updateChoiceStyles(); drawFrame(currentFrame); });
    buildChoices(rightChoices, "right", options, (value) => { selectedRight = value; updateChoiceStyles(); drawFrame(currentFrame); });
    frameSlider.max = Math.max(bundle.frames.length - 1, 0);
    frameSlider.oninput = (event) => drawFrame(Number(event.target.value));
    playToggle.onclick = togglePlayback;
    submitButton.onclick = submitReview;
    renderCards();
    renderMergeQuestions();
    drawFrame(0);
  }

  boot();
}
