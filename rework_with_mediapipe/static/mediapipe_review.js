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
  const titleNode = document.getElementById("mpr-title");
  const subtitleNode = document.getElementById("mpr-subtitle");
  const frameImage = document.getElementById("mpr-frame-image");
  const overlayCanvas = document.getElementById("mpr-overlay-canvas");
  const prevKeyframeButton = document.getElementById("mpr-prev-keyframe");
  const nextKeyframeButton = document.getElementById("mpr-next-keyframe");
  const keyframeLabel = document.getElementById("mpr-keyframe-label");
  const keyframeReasons = document.getElementById("mpr-keyframe-reasons");
  const keyframeStatus = document.getElementById("mpr-keyframe-status");
  const viewerHint = document.getElementById("mpr-viewer-hint");
  const leftMissingBox = document.getElementById("mpr-left-missing-box");
  const rightMissingBox = document.getElementById("mpr-right-missing-box");
  const clearLeftButton = document.getElementById("mpr-clear-left");
  const clearRightButton = document.getElementById("mpr-clear-right");
  const confirmButton = document.getElementById("mpr-confirm-keyframe");
  const assignmentSummary = document.getElementById("mpr-assignment-summary");
  const keyframeList = document.getElementById("mpr-keyframe-list");
  const stripSummary = document.getElementById("mpr-strip-summary");
  const submitButton = document.getElementById("mpr-submit-review");
  const submitMessage = document.getElementById("mpr-submit-message");

  let bundle = null;
  let currentKeyframeIndex = 0;
  let drawnBoxes = [];

  const frameTracksByFrameIdx = new Map();
  const keyframeReviewByFrame = new Map();

  function getKeyframes() {
    return bundle?.keyframes || [];
  }

  function getCurrentFrame() {
    return getCurrentKeyframe() || null;
  }

  function getCurrentKeyframe() {
    return getKeyframes()[currentKeyframeIndex] || null;
  }

  function ensureReviewEntry(keyframe) {
    if (!keyframe) return null;
    const frameIdx = Number(keyframe.frame_idx);
    let review = keyframeReviewByFrame.get(frameIdx);
    if (!review) {
      review = {
        frame_idx: frameIdx,
        confirmed: false,
        left_track_id: null,
        right_track_id: null,
        left_missing_box: false,
        right_missing_box: false,
      };
      keyframeReviewByFrame.set(frameIdx, review);
    }
    return review;
  }

  function getCurrentReview() {
    return ensureReviewEntry(getCurrentKeyframe());
  }

  function visibleTracksForFrame(frameIdx) {
    return frameTracksByFrameIdx.get(Number(frameIdx))?.tracks || [];
  }

  function roleForTrack(trackId) {
    const review = getCurrentReview();
    if (!review) return "neutral";
    if (review.left_track_id != null && Number(review.left_track_id) === Number(trackId)) return "left";
    if (review.right_track_id != null && Number(review.right_track_id) === Number(trackId)) return "right";
    return "neutral";
  }

  function frameRoleSummary(review) {
    if (!review) return "未初始化";
    const leftLabel = review.left_missing_box
      ? "左手没框"
      : review.left_track_id == null
        ? "左手无保留"
        : `左手=T${review.left_track_id}`;
    const rightLabel = review.right_missing_box
      ? "右手没框"
      : review.right_track_id == null
        ? "右手无保留"
        : `右手=T${review.right_track_id}`;
    const confirmLabel = review.confirmed ? "已确认" : "未确认";
    return `${leftLabel} · ${rightLabel} · ${confirmLabel}`;
  }

  function allConfirmed() {
    return getKeyframes().every((keyframe) => ensureReviewEntry(keyframe)?.confirmed);
  }

  function renderCurrentKeyframeImage() {
    const frame = getCurrentFrame();
    if (!frame) return;
    const src = `/mediapipe/assets/${frame.relpath}`;
    if (frameImage.dataset.src === src && frameImage.complete) {
      drawOverlay();
      return;
    }
    frameImage.dataset.src = src;
    frameImage.onload = () => {
      drawOverlay();
    };
    frameImage.src = src;
  }

  function selectKeyframe(index) {
    const keyframes = getKeyframes();
    if (!keyframes.length) return;
    currentKeyframeIndex = Math.max(0, Math.min(index, keyframes.length - 1));
    renderCurrentKeyframeImage();
    renderReviewState();
  }

  function assignmentForSide(side) {
    const review = getCurrentReview();
    if (!review) return;
    review.confirmed = false;
    if (side === "left") {
      review.left_track_id = null;
      review.left_missing_box = false;
    } else {
      review.right_track_id = null;
      review.right_missing_box = false;
    }
    renderReviewState();
    drawOverlay();
  }

  function setMissingBox(side, checked) {
    const review = getCurrentReview();
    if (!review) return;
    review.confirmed = false;
    if (side === "left") {
      review.left_missing_box = checked;
      if (checked) review.left_track_id = null;
    } else {
      review.right_missing_box = checked;
      if (checked) review.right_track_id = null;
    }
    renderReviewState();
    drawOverlay();
  }

  function assignTrack(side, trackId) {
    const review = getCurrentReview();
    if (!review) return;
    review.confirmed = false;
    if (side === "left") {
      review.left_missing_box = false;
      review.left_track_id = review.left_track_id != null && Number(review.left_track_id) === Number(trackId) ? null : Number(trackId);
      if (review.right_track_id != null && Number(review.right_track_id) === Number(trackId)) review.right_track_id = null;
    } else {
      review.right_missing_box = false;
      review.right_track_id = review.right_track_id != null && Number(review.right_track_id) === Number(trackId) ? null : Number(trackId);
      if (review.left_track_id != null && Number(review.left_track_id) === Number(trackId)) review.left_track_id = null;
    }
    renderReviewState();
    drawOverlay();
  }

  function confirmCurrentKeyframe() {
    const review = getCurrentReview();
    if (!review) return;
    review.confirmed = true;
    renderReviewState();
  }

  function renderReviewState() {
    const keyframe = getCurrentKeyframe();
    const review = getCurrentReview();
    if (!keyframe || !review) return;
    const reasons = (keyframe.reasons || []).join(" · ") || "无特殊原因";
    const segmentFrames = (bundle.segments || []).find((item) => Number(item.segment_id) === Number(keyframe.segment_id));
    keyframeLabel.textContent = `关键帧 ${currentKeyframeIndex + 1}/${getKeyframes().length} · frame ${keyframe.frame_idx}`;
    keyframeReasons.textContent = `原因: ${reasons}${segmentFrames ? ` · 段 ${segmentFrames.start_frame}-${segmentFrames.end_frame}` : ""}`;
    keyframeStatus.textContent = review.confirmed ? "已确认" : "待确认";
    leftMissingBox.checked = Boolean(review.left_missing_box);
    rightMissingBox.checked = Boolean(review.right_missing_box);
    confirmButton.textContent = review.confirmed ? "重新确认当前关键帧" : "确认当前关键帧";
    assignmentSummary.textContent = frameRoleSummary(review);
    viewerHint.textContent = "当前页面只显示关键帧。左键=左手，右键=右手，再点一次同侧可清空。";
    stripSummary.textContent = `${getKeyframes().filter((item) => ensureReviewEntry(item)?.confirmed).length}/${getKeyframes().length} 已确认`;
    submitButton.disabled = !allConfirmed();
    prevKeyframeButton.disabled = currentKeyframeIndex <= 0;
    nextKeyframeButton.disabled = currentKeyframeIndex >= getKeyframes().length - 1;
    renderKeyframeList();
  }

  function renderKeyframeList() {
    keyframeList.innerHTML = "";
    getKeyframes().forEach((keyframe, index) => {
      const review = ensureReviewEntry(keyframe);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "mpr-keyframe-chip";
      button.classList.add(review?.confirmed ? "is-confirmed" : "is-pending");
      if (index === currentKeyframeIndex) button.classList.add("is-current");
      const reasonText = (keyframe.reasons || []).join(" · ") || "普通覆盖";
      button.innerHTML = `
        <div class="mpr-chip-topline">
          <span>frame ${keyframe.frame_idx}</span>
          <span class="mpr-chip-kind">${keyframe.kind === "anchor" ? "anchor" : "coverage"}</span>
        </div>
        <div class="mpr-chip-reasons">${reasonText}</div>
        <div class="mpr-chip-state">${frameRoleSummary(review)}</div>
      `;
      button.onclick = () => {
        selectKeyframe(index);
      };
      keyframeList.appendChild(button);
    });
  }

  function drawOverlay() {
    const frame = getCurrentFrame();
    if (!frame || !frameImage.complete) return;
    overlayCanvas.width = frameImage.clientWidth;
    overlayCanvas.height = frameImage.clientHeight;
    const ctx = overlayCanvas.getContext("2d");
    const naturalWidth = frameImage.naturalWidth || 1;
    const naturalHeight = frameImage.naturalHeight || 1;
    const sx = overlayCanvas.width / naturalWidth;
    const sy = overlayCanvas.height / naturalHeight;
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    drawnBoxes = [];
    visibleTracksForFrame(frame.frame_idx).forEach((track) => {
      const trackId = Number(track.track_id);
      const role = roleForTrack(trackId);
      const color = role === "left" ? "#0f766e" : role === "right" ? "#b91c1c" : "#2563eb";
      const label = role === "left" ? `L · T${trackId}` : role === "right" ? `R · T${trackId}` : `T${trackId}`;
      const [x1, y1, x2, y2] = track.bbox_xyxy;
      const left = x1 * sx;
      const top = y1 * sy;
      const width = (x2 - x1) * sx;
      const height = (y2 - y1) * sy;
      ctx.strokeStyle = color;
      ctx.lineWidth = role === "neutral" ? 3 : 5;
      ctx.strokeRect(left, top, width, height);
      ctx.font = "bold 15px sans-serif";
      const textWidth = ctx.measureText(label).width;
      const labelX = left;
      const labelY = Math.max(24, top);
      ctx.fillStyle = color;
      ctx.fillRect(labelX, labelY - 20, textWidth + 16, 24);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, labelX + 8, labelY - 4);
      drawnBoxes.push({
        track_id: trackId,
        left,
        top,
        right: left + width,
        bottom: top + height,
        area: width * height,
      });
    });
  }

  function buildPayload() {
    return {
      review_version: "keyframe_v1",
      keyframe_reviews: getKeyframes().map((keyframe) => {
        const review = ensureReviewEntry(keyframe);
        return {
          frame_idx: Number(review.frame_idx),
          confirmed: Boolean(review.confirmed),
          left_track_id: review.left_track_id == null ? null : Number(review.left_track_id),
          right_track_id: review.right_track_id == null ? null : Number(review.right_track_id),
          left_missing_box: Boolean(review.left_missing_box),
          right_missing_box: Boolean(review.right_missing_box),
        };
      }),
    };
  }

  async function submitReview() {
    if (!allConfirmed()) {
      submitMessage.textContent = "还有关键帧未确认，不能提交。";
      return;
    }
    const response = await fetch(`/api/mediapipe/clips/${clipId}/submit-review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
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

  function hydrateFromExistingReview(reviewPayload) {
    if (!reviewPayload || typeof reviewPayload !== "object") return;
    if (String(reviewPayload.review_version || "") !== "keyframe_v1") return;
    (reviewPayload.keyframe_reviews || []).forEach((item) => {
      const frameIdx = Number(item.frame_idx);
      keyframeReviewByFrame.set(frameIdx, {
        frame_idx: frameIdx,
        confirmed: Boolean(item.confirmed),
        left_track_id: item.left_track_id == null ? null : Number(item.left_track_id),
        right_track_id: item.right_track_id == null ? null : Number(item.right_track_id),
        left_missing_box: Boolean(item.left_missing_box),
        right_missing_box: Boolean(item.right_missing_box),
      });
    });
  }

  overlayCanvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });

  overlayCanvas.addEventListener("mousedown", (event) => {
    if (event.button !== 0 && event.button !== 2) return;
    event.preventDefault();
    const rect = overlayCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const candidates = drawnBoxes
      .filter((item) => x >= item.left && x <= item.right && y >= item.top && y <= item.bottom)
      .sort((a, b) => a.area - b.area);
    if (!candidates.length) return;
    assignTrack(event.button === 0 ? "left" : "right", candidates[0].track_id);
  });

  leftMissingBox.onchange = () => setMissingBox("left", leftMissingBox.checked);
  rightMissingBox.onchange = () => setMissingBox("right", rightMissingBox.checked);
  clearLeftButton.onclick = () => assignmentForSide("left");
  clearRightButton.onclick = () => assignmentForSide("right");
  confirmButton.onclick = confirmCurrentKeyframe;
  prevKeyframeButton.onclick = () => {
    selectKeyframe(currentKeyframeIndex - 1);
  };
  nextKeyframeButton.onclick = () => {
    selectKeyframe(currentKeyframeIndex + 1);
  };
  submitButton.onclick = submitReview;

  async function boot() {
    const response = await fetch(`/api/mediapipe/clips/${clipId}`);
    const data = await response.json();
    bundle = data.bundle;
    if (!bundle) {
      titleNode.textContent = "这个 clip 还没有可用的 bundle。";
      return;
    }
    if (!getKeyframes().length) {
      titleNode.textContent = "这个 clip 没有可审核的关键帧。";
      return;
    }
    (bundle.frame_tracks || []).forEach((item) => {
      frameTracksByFrameIdx.set(Number(item.frame_idx), item);
    });
    titleNode.textContent = `${bundle.episode_name} · frames ${bundle.clip_start}-${bundle.clip_end}`;
    subtitleNode.textContent = `${bundle.dirty_reason} · ${getKeyframes().length} 个关键帧 · ${(bundle.segments || []).length} 个稳定段`;
    hydrateFromExistingReview(data.clip?.review_payload_json);
    const firstPending = getKeyframes().findIndex((keyframe) => !ensureReviewEntry(keyframe)?.confirmed);
    currentKeyframeIndex = firstPending >= 0 ? firstPending : 0;
    selectKeyframe(currentKeyframeIndex);
  }

  boot().catch((error) => {
    titleNode.textContent = String(error);
  });
}
