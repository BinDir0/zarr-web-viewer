const dashboardNode = document.getElementById("mpr-dashboard");
const reviewNode = document.getElementById("mpr-review-app");

function renderSummary(summary) {
  const countsNode = document.getElementById("summaryCounts");
  const startLink = document.getElementById("mpr-start-link");
  const startEmpty = document.getElementById("mpr-start-empty");
  const startActions = document.getElementById("mpr-start-actions");
  if (countsNode && summary.counts) {
    countsNode.innerHTML = Object.entries(summary.counts)
      .map(
        ([key, value]) => `
        <div class="mpr-count-item">
          <div class="mpr-count-key">${key}</div>
          <div class="mpr-count-value">${value}</div>
        </div>`
      )
      .join("");
  }
  if (startActions) {
    if (summary.next_clip_id) {
      if (startLink) {
        startLink.href = `/mediapipe/clip/${summary.next_clip_id}`;
        startLink.textContent = `开始审核 Clip ${summary.next_clip_id}`;
        startLink.style.display = "";
      }
      if (startEmpty) startEmpty.style.display = "none";
    } else {
      if (startLink) startLink.style.display = "none";
      if (startEmpty) startEmpty.style.display = "";
    }
  }
}

async function fetchSummary() {
  const response = await fetch("/api/mediapipe/summary");
  const data = await response.json();
  if (response.ok && data.success) {
    renderSummary(data);
  }
}

if (dashboardNode) {
  fetchSummary().catch((error) => {
    console.error(error);
  });
}

if (reviewNode) {
  const clipId = Number(reviewNode.dataset.clipId);
  const frameImage = document.getElementById("mpr-frame-image");
  const overlayCanvas = document.getElementById("mpr-overlay-canvas");
  const frameSlider = document.getElementById("mpr-frame-slider");
  const frameLabel = document.getElementById("mpr-frame-label");
  const playToggle = document.getElementById("mpr-play-toggle");
  const submitButton = document.getElementById("mpr-submit-review");
  const submitMessage = document.getElementById("mpr-submit-message");
  const candidateCards = document.getElementById("mpr-candidate-cards");
  const titleNode = document.getElementById("mpr-title");
  const leftSummary = document.getElementById("mpr-left-summary");
  const rightSummary = document.getElementById("mpr-right-summary");
  const leftMissingBox = document.getElementById("mpr-left-missing-box");
  const rightMissingBox = document.getElementById("mpr-right-missing-box");
  const leftClear = document.getElementById("mpr-left-clear");
  const rightClear = document.getElementById("mpr-right-clear");
  const badFrameChips = document.getElementById("mpr-bad-frame-chips");
  const focusHint = document.getElementById("mpr-focus-hint");

  let bundle = null;
  let currentFrame = 0;
  let playing = false;
  let timerId = null;
  const selectedLeft = new Set();
  const selectedRight = new Set();

  function trackRole(trackId) {
    if (selectedLeft.has(trackId)) return "left";
    if (selectedRight.has(trackId)) return "right";
    return "ignore";
  }

  function setTrackRole(trackId, role) {
    selectedLeft.delete(trackId);
    selectedRight.delete(trackId);
    if (role === "left") selectedLeft.add(trackId);
    if (role === "right") selectedRight.add(trackId);
    renderSideSummaries();
    renderCards();
    drawFrame(currentFrame);
  }

  function sortedIds(setLike) {
    return Array.from(setLike.values()).sort((a, b) => a - b);
  }

  function summaryText(side) {
    const ids = side === "left" ? sortedIds(selectedLeft) : sortedIds(selectedRight);
    const missing = side === "left" ? leftMissingBox.checked : rightMissingBox.checked;
    if (missing && ids.length > 0) {
      return `已选 Track ${ids.join(", ")}；并标记该侧仍有可见手缺框。`;
    }
    if (missing) {
      return "该侧手可见，但当前 proposal 没有把框打出来。";
    }
    if (ids.length > 0) {
      return `已归到该侧的 Track: ${ids.join(", ")}。`;
    }
    return "当前未选任何 Track，默认表示该侧没出现 wearer hand。";
  }

  function renderSideSummaries() {
    leftSummary.textContent = summaryText("left");
    rightSummary.textContent = summaryText("right");
  }

  function renderBadFrames() {
    badFrameChips.innerHTML = "";
    const badFrames = bundle.bad_frame_indices || [];
    if (badFrames.length === 0) {
      focusHint.textContent = "这条 clip 没有显式 bad frame 标记，可按整体观看。";
      return;
    }
    focusHint.textContent = `重点检查 bad frames: ${badFrames.join(", ")}`;
    badFrames.forEach((frameIdx) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "mpr-btn mpr-btn-secondary mpr-bad-chip";
      chip.textContent = `Frame ${frameIdx}`;
      chip.onclick = () => {
        const targetIndex = bundle.frames.findIndex((item) => item.frame_idx === frameIdx);
        if (targetIndex >= 0) drawFrame(targetIndex);
      };
      badFrameChips.appendChild(chip);
    });
  }

  function drawFrame(frameIndex) {
    if (!bundle) return;
    const frame = bundle.frames[frameIndex];
    if (!frame) return;
    currentFrame = frameIndex;
    frameImage.src = `/mediapipe/assets/${frame.relpath}`;
    frameLabel.textContent = `${frameIndex + 1} / ${bundle.frames.length} · frame ${frame.frame_idx}${frame.is_bad ? " · BAD" : ""}`;
    frameSlider.value = frameIndex;
    frameImage.onload = () => {
      overlayCanvas.width = frameImage.clientWidth;
      overlayCanvas.height = frameImage.clientHeight;
      const ctx = overlayCanvas.getContext("2d");
      const naturalWidth = frameImage.naturalWidth || 1;
      const naturalHeight = frameImage.naturalHeight || 1;
      const sx = overlayCanvas.width / naturalWidth;
      const sy = overlayCanvas.height / naturalHeight;
      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      bundle.tracks.forEach((track) => {
        const proposal = track.proposals.find((item) => item.frame_idx === frame.frame_idx);
        if (!proposal) return;
        const trackId = Number(track.track_id);
        const role = trackRole(trackId);
        const color = role === "left" ? "#0f766e" : role === "right" ? "#b91c1c" : "#2563eb";
        const label = role === "left" ? `L · Track ${trackId}` : role === "right" ? `R · Track ${trackId}` : `Track ${trackId}`;
        const [x1, y1, x2, y2] = proposal.bbox_xyxy;
        ctx.strokeStyle = color;
        ctx.lineWidth = role === "ignore" ? 3 : 4;
        ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
        ctx.font = "bold 14px sans-serif";
        const textWidth = ctx.measureText(label).width;
        const labelX = x1 * sx;
        const labelY = Math.max(22, y1 * sy);
        ctx.fillStyle = color;
        ctx.fillRect(labelX, labelY - 18, textWidth + 14, 22);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, labelX + 7, labelY - 3);
      });
    };
  }

  function renderCards() {
    candidateCards.innerHTML = "";
    const cards = bundle.track_cards || [];
    if (cards.length === 0) {
      candidateCards.innerHTML = `<div class="mpr-meta-note">当前 clip 没有任何候选 Track。若某侧确实可见，请勾选“该侧可见但缺框”。</div>`;
      return;
    }
    cards.forEach((card) => {
      const trackId = Number(card.track_id);
      const role = trackRole(trackId);
      const node = document.createElement("article");
      node.className = "mpr-candidate-card";
      if (role === "left") node.classList.add("is-left-selected");
      if (role === "right") node.classList.add("is-right-selected");
      node.innerHTML = `
        <img src="/mediapipe/assets/${card.preview_relpath}" alt="track preview">
        <div class="mpr-candidate-head">
          <h3>Track ${trackId}</h3>
          <button class="mpr-btn mpr-btn-secondary mpr-jump-btn" type="button">跳到参考帧</button>
        </div>
        <div class="mpr-inline">
          ${card.intersects_bad_frames ? '<span class="mpr-badge bad">触及 bad frame</span>' : '<span class="mpr-badge">上下文候选</span>'}
          <span class="mpr-badge">${card.role_hint}</span>
        </div>
        <div class="mpr-meta-note">来源: MediaPipe VIDEO · ${card.handedness_label} · slot ${card.side_rank}</div>
        <div class="mpr-meta-note">frames ${card.start_frame}-${card.end_frame} · ${card.num_frames} 帧</div>
        <div class="mpr-inline">
          <button class="mpr-btn mpr-choice-btn mpr-quick-left" type="button">归到左手</button>
          <button class="mpr-btn mpr-choice-btn mpr-quick-right" type="button">归到右手</button>
          <button class="mpr-btn mpr-btn-secondary mpr-quick-ignore" type="button">忽略</button>
        </div>
      `;
      node.querySelector(".mpr-jump-btn").onclick = () => {
        const targetIndex = bundle.frames.findIndex((item) => item.frame_idx === card.preview_frame);
        if (targetIndex >= 0) drawFrame(targetIndex);
      };
      node.querySelector(".mpr-quick-left").onclick = () => setTrackRole(trackId, "left");
      node.querySelector(".mpr-quick-right").onclick = () => setTrackRole(trackId, "right");
      node.querySelector(".mpr-quick-ignore").onclick = () => setTrackRole(trackId, "ignore");
      candidateCards.appendChild(node);
    });
  }

  async function submitReview() {
    const payload = {
      left_track_ids: sortedIds(selectedLeft),
      right_track_ids: sortedIds(selectedRight),
      left_missing_box: leftMissingBox.checked,
      right_missing_box: rightMissingBox.checked,
    };
    const response = await fetch(`/api/mediapipe/clips/${clipId}/submit-review`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok || !data.success) {
      submitMessage.textContent = data.message || "提交失败";
      return;
    }
    if (data.next_clip_id) {
      window.location.href = `/mediapipe/clip/${data.next_clip_id}`;
      return;
    }
    submitMessage.textContent = "提交成功，没有更多 ready clip。";
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
    }, 140);
  }

  function hydrateFromExistingReview(reviewPayload) {
    if (!reviewPayload || typeof reviewPayload !== "object") return;
    (reviewPayload.left_track_ids || []).forEach((item) => selectedLeft.add(Number(item)));
    (reviewPayload.right_track_ids || []).forEach((item) => selectedRight.add(Number(item)));
    leftMissingBox.checked = Boolean(reviewPayload.left_missing_box);
    rightMissingBox.checked = Boolean(reviewPayload.right_missing_box);
  }

  async function boot() {
    const response = await fetch(`/api/mediapipe/clips/${clipId}`);
    const data = await response.json();
    bundle = data.bundle;
    if (!bundle) {
      titleNode.textContent = "这个 clip 还没有可用的 bundle。";
      return;
    }
    titleNode.textContent = `${bundle.episode_name} · frames ${bundle.clip_start}-${bundle.clip_end}`;
    hydrateFromExistingReview(data.clip?.review_payload_json);
    leftMissingBox.onchange = renderSideSummaries;
    rightMissingBox.onchange = renderSideSummaries;
    leftClear.onclick = () => {
      selectedLeft.clear();
      renderSideSummaries();
      renderCards();
      drawFrame(currentFrame);
    };
    rightClear.onclick = () => {
      selectedRight.clear();
      renderSideSummaries();
      renderCards();
      drawFrame(currentFrame);
    };
    frameSlider.max = Math.max(bundle.frames.length - 1, 0);
    frameSlider.oninput = (event) => drawFrame(Number(event.target.value));
    playToggle.onclick = togglePlayback;
    submitButton.onclick = submitReview;
    renderBadFrames();
    renderSideSummaries();
    renderCards();
    const firstBadFrame = (bundle.bad_frame_indices || [])[0];
    const firstIndex = firstBadFrame === undefined
      ? 0
      : Math.max(0, bundle.frames.findIndex((item) => item.frame_idx === firstBadFrame));
    drawFrame(firstIndex);
  }

  boot().catch((error) => {
    titleNode.textContent = String(error);
  });
}
