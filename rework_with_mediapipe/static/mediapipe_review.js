const dashboardNode = document.getElementById("mpr-dashboard");
const reviewNode = document.getElementById("mpr-review-app");

function renderSummary(summary) {
  const countsNode = document.getElementById("summaryCounts");
  const startLink = document.getElementById("mpr-start-link");
  const startEmpty = document.getElementById("mpr-start-empty");
  const startActions = document.getElementById("mpr-start-actions");
  const startRankInput = document.getElementById("mpr-start-rank");
  const startRank = Math.max(1, Number(startRankInput?.value || summary.requested_start_rank || 1));
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
        startLink.href = `/mediapipe/clip/${summary.next_clip_id}?start_rank=${startRank}`;
        startLink.textContent = `打开第 ${startRank} 条 Episode ${summary.next_clip_id}`;
        startLink.style.display = "";
      }
      if (startEmpty) startEmpty.style.display = "none";
    } else {
      if (startLink) startLink.style.display = "none";
      if (startEmpty) startEmpty.style.display = "";
      if (startEmpty) {
        startEmpty.textContent = `从第 ${startRank} 条开始时，当前没有已预处理 episode`;
      }
    }
  }
}

async function fetchSummary(startRank = 1) {
  const rank = Math.max(1, Number(startRank || 1));
  const response = await fetch(`/api/mediapipe/summary?start_rank=${encodeURIComponent(rank)}`);
  const data = await response.json();
  if (response.ok && data.success) {
    renderSummary(data);
  }
}

if (dashboardNode) {
  const startRankInput = document.getElementById("mpr-start-rank");
  const refreshSummary = () => {
    const startRank = Math.max(1, Number(startRankInput?.value || 1));
    if (startRankInput) {
      startRankInput.value = String(startRank);
    }
    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.set("start_rank", String(startRank));
    window.history.replaceState({}, "", nextUrl);
    return fetchSummary(startRank);
  };
  if (startRankInput) {
    startRankInput.addEventListener("input", () => {
      refreshSummary().catch((error) => {
        console.error(error);
      });
    });
    startRankInput.addEventListener("change", () => {
      refreshSummary().catch((error) => {
        console.error(error);
      });
    });
  }
  refreshSummary().catch((error) => {
    console.error(error);
  });
}

if (reviewNode) {
  const clipId = Number(reviewNode.dataset.clipId);
  const startRank = Math.max(1, Number(reviewNode.dataset.startRank || 1));
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
  const leftMissingButton = document.getElementById("mpr-left-missing-box");
  const rightMissingButton = document.getElementById("mpr-right-missing-box");
  const leftNotVisibleButton = document.getElementById("mpr-left-not-visible");
  const rightNotVisibleButton = document.getElementById("mpr-right-not-visible");
  const assignmentSummary = document.getElementById("mpr-assignment-summary");
  const stripSummary = document.getElementById("mpr-strip-summary");
  const jumpInput = document.getElementById("mpr-jump-input");
  const jumpGoButton = document.getElementById("mpr-jump-go");
  const submitButton = document.getElementById("mpr-submit-review");
  const submitMessage = document.getElementById("mpr-submit-message");

  let bundle = null;
  let currentReviewFrameIndex = 0;
  let drawnBoxes = [];
  let cachedReviewFrames = null;
  const imagePreloadCache = new Map();

  const frameTracksByFrameIdx = new Map();
  const frameMetaByFrameIdx = new Map();
  const frameReviewByFrame = new Map();

  function getBaseKeyframes() {
    return bundle?.keyframes || [];
  }

  function invalidateReviewFrames() {
    cachedReviewFrames = null;
  }

  function frameSegmentId(frameIdx) {
    return Number(frameMetaByFrameIdx.get(Number(frameIdx))?.segment_id ?? -1);
  }

  function ensureReviewEntry(frame) {
    if (!frame) return null;
    const frameIdx = Number(frame.frame_idx);
    let review = frameReviewByFrame.get(frameIdx);
    if (!review) {
      review = {
        frame_idx: frameIdx,
        confirmed: false,
        left_mode: "not_visible",
        right_mode: "not_visible",
        left_track_id: null,
        right_track_id: null,
      };
      frameReviewByFrame.set(frameIdx, review);
    }
    return review;
  }

  function segmentNeedsRecovery(segmentId) {
    for (const review of frameReviewByFrame.values()) {
      if (frameSegmentId(review.frame_idx) !== Number(segmentId)) continue;
      if (review.left_mode === "visible_unrecoverable" || review.right_mode === "visible_unrecoverable") return true;
    }
    return false;
  }

  function buildRecoveryFrame(frameIdx, segmentId) {
    const frameMeta = frameMetaByFrameIdx.get(Number(frameIdx));
    if (!frameMeta) return null;
    return {
      frame_idx: Number(frameIdx),
      relpath: frameMeta.relpath,
      segment_id: Number(segmentId),
      kind: "recovery",
      reasons: ["recovery_anchor"],
      visible_track_ids: frameMeta.visible_track_ids || [],
      is_recovery_frame: true,
    };
  }

  function getReviewFrames() {
    if (!bundle) return [];
    if (cachedReviewFrames) return cachedReviewFrames;
    const merged = new Map();
    getBaseKeyframes().forEach((item) => {
      merged.set(Number(item.frame_idx), {
        ...item,
        is_recovery_frame: false,
      });
    });
    for (const segment of bundle.segments || []) {
      const segmentId = Number(segment.segment_id);
      if (!segmentNeedsRecovery(segmentId)) continue;
      for (const frameIdx of segment.recovery_candidate_frames || []) {
        if (merged.has(Number(frameIdx))) continue;
        const recoveryFrame = buildRecoveryFrame(frameIdx, segmentId);
        if (recoveryFrame) merged.set(Number(frameIdx), recoveryFrame);
      }
    }
    cachedReviewFrames = Array.from(merged.values()).sort((a, b) => Number(a.frame_idx) - Number(b.frame_idx));
    return cachedReviewFrames;
  }

  function ensureCurrentReviewFrame(preferredFrameIdx = null) {
    const reviewFrames = getReviewFrames();
    if (!reviewFrames.length) {
      currentReviewFrameIndex = 0;
      return;
    }
    const currentFrameIdx = preferredFrameIdx != null
      ? Number(preferredFrameIdx)
      : Number(reviewFrames[currentReviewFrameIndex]?.frame_idx);
    const nextIndex = reviewFrames.findIndex((item) => Number(item.frame_idx) === currentFrameIdx);
    if (nextIndex >= 0) {
      currentReviewFrameIndex = nextIndex;
      return;
    }
    currentReviewFrameIndex = Math.max(0, Math.min(currentReviewFrameIndex, reviewFrames.length - 1));
  }

  function getCurrentFrame() {
    ensureCurrentReviewFrame();
    return getReviewFrames()[currentReviewFrameIndex] || null;
  }

  function getCurrentReview() {
    return ensureReviewEntry(getCurrentFrame());
  }

  function markReviewConfirmed(frame = getCurrentFrame()) {
    const review = ensureReviewEntry(frame);
    if (!review) return;
    review.confirmed = true;
  }

  function visibleTracksForFrame(frameIdx) {
    return frameTracksByFrameIdx.get(Number(frameIdx))?.tracks || [];
  }

  function roleForTrack(trackId) {
    const review = getCurrentReview();
    if (!review) return "neutral";
    if (review.left_mode === "track" && Number(review.left_track_id) === Number(trackId)) return "left";
    if (review.right_mode === "track" && Number(review.right_track_id) === Number(trackId)) return "right";
    return "neutral";
  }

  function sideSummary(review, side) {
    const prefix = side === "left" ? "左手" : "右手";
    const mode = review?.[`${side}_mode`] || "not_visible";
    if (mode === "track") return `${prefix}=T${review[`${side}_track_id`]}`;
    if (mode === "visible_unrecoverable") return `${prefix}=无法恢复`;
    return `${prefix}=不可见`;
  }

  function frameRoleSummary(review) {
    if (!review) return "未初始化";
    const confirmLabel = review.confirmed ? "已确认" : "未确认";
    return `${sideSummary(review, "left")} · ${sideSummary(review, "right")} · ${confirmLabel}`;
  }

  function allConfirmed() {
    return getReviewFrames().every((frame) => ensureReviewEntry(frame)?.confirmed);
  }

  function frameAssetUrl(frame) {
    return `/mediapipe/assets/${frame.relpath}`;
  }

  function ensureImagePreloaded(frame) {
    if (!frame?.relpath) return null;
    const src = frameAssetUrl(frame);
    let img = imagePreloadCache.get(src);
    if (img) return img;
    img = new Image();
    img.decoding = "async";
    img.src = src;
    imagePreloadCache.set(src, img);
    return img;
  }

  function warmImageWindow(centerIndex, radius = 3) {
    const reviewFrames = getReviewFrames();
    const start = Math.max(0, centerIndex - radius);
    const end = Math.min(reviewFrames.length - 1, centerIndex + radius);
    for (let index = start; index <= end; index += 1) {
      ensureImagePreloaded(reviewFrames[index]);
    }
  }

  function renderCurrentFrameImage() {
    const frame = getCurrentFrame();
    if (!frame) return;
    const src = frameAssetUrl(frame);
    ensureImagePreloaded(frame);
    warmImageWindow(currentReviewFrameIndex);
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

  function selectReviewFrame(index) {
    const reviewFrames = getReviewFrames();
    if (!reviewFrames.length) return;
    const currentFrame = getCurrentFrame();
    if (currentFrame) {
      markReviewConfirmed(currentFrame);
    }
    currentReviewFrameIndex = Math.max(0, Math.min(index, reviewFrames.length - 1));
    warmImageWindow(currentReviewFrameIndex);
    renderCurrentFrameImage();
    renderReviewState();
  }

  function setSideNotVisible(side) {
    const review = getCurrentReview();
    if (!review) return;
    const currentFrameIdx = review.frame_idx;
    review[`${side}_mode`] = "not_visible";
    review[`${side}_track_id`] = null;
    review.confirmed = true;
    invalidateReviewFrames();
    ensureCurrentReviewFrame(currentFrameIdx);
    renderReviewState();
    drawOverlay();
  }

  function toggleUnrecoverable(side) {
    const review = getCurrentReview();
    if (!review) return;
    const currentFrameIdx = review.frame_idx;
    if (review[`${side}_mode`] === "visible_unrecoverable") {
      review[`${side}_mode`] = "not_visible";
    } else {
      review[`${side}_mode`] = "visible_unrecoverable";
      review[`${side}_track_id`] = null;
    }
    review.confirmed = true;
    invalidateReviewFrames();
    ensureCurrentReviewFrame(currentFrameIdx);
    renderReviewState();
    drawOverlay();
  }

  function assignTrack(side, trackId) {
    const review = getCurrentReview();
    if (!review) return;
    const currentFrameIdx = review.frame_idx;
    if (review[`${side}_mode`] === "track" && Number(review[`${side}_track_id`]) === Number(trackId)) {
      review[`${side}_mode`] = "not_visible";
      review[`${side}_track_id`] = null;
    } else {
      review[`${side}_mode`] = "track";
      review[`${side}_track_id`] = Number(trackId);
    }
    const otherSide = side === "left" ? "right" : "left";
    if (review[`${otherSide}_mode`] === "track" && Number(review[`${otherSide}_track_id`]) === Number(trackId)) {
      review[`${otherSide}_mode`] = "not_visible";
      review[`${otherSide}_track_id`] = null;
    }
    review.confirmed = true;
    invalidateReviewFrames();
    ensureCurrentReviewFrame(currentFrameIdx);
    renderReviewState();
    drawOverlay();
  }

  function renderReviewState() {
    ensureCurrentReviewFrame();
    const reviewFrame = getCurrentFrame();
    const review = getCurrentReview();
    if (!reviewFrame || !review) return;
    const reviewFrames = getReviewFrames();
    const reasons = (reviewFrame.reasons || []).join(" · ") || "无特殊原因";
    const segment = (bundle.segments || []).find((item) => Number(item.segment_id) === Number(reviewFrame.segment_id));
    keyframeLabel.textContent = `审核帧 ${currentReviewFrameIndex + 1}/${reviewFrames.length} · frame ${reviewFrame.frame_idx}`;
    keyframeReasons.textContent = `原因: ${reasons}${segment ? ` · 段 ${segment.start_frame}-${segment.end_frame}` : ""}`;
    keyframeStatus.textContent = review.confirmed ? "已确认" : "待确认";
    leftMissingButton.classList.toggle("is-active", review.left_mode === "visible_unrecoverable");
    rightMissingButton.classList.toggle("is-active", review.right_mode === "visible_unrecoverable");
    leftNotVisibleButton.classList.toggle("is-active", review.left_mode === "not_visible");
    rightNotVisibleButton.classList.toggle("is-active", review.right_mode === "not_visible");
    assignmentSummary.textContent = frameRoleSummary(review);
    viewerHint.textContent = "左键点框=左手，右键点框=右手；下方按钮可快速标记不可见/可见但缺框；←/→ 切换审核帧。";
    submitButton.disabled = !allConfirmed();
    prevKeyframeButton.disabled = currentReviewFrameIndex <= 0;
    nextKeyframeButton.disabled = currentReviewFrameIndex >= reviewFrames.length - 1;
    renderReviewFrameList();
  }

  function renderReviewFrameList() {
    const reviewFrames = getReviewFrames();
    const total = reviewFrames.length;
    const confirmedCount = reviewFrames.filter((item) => ensureReviewEntry(item)?.confirmed).length;
    stripSummary.textContent = `${confirmedCount}/${total} 已确认`;
    if (jumpInput) {
      jumpInput.min = total > 0 ? "1" : "0";
      jumpInput.max = String(Math.max(total, 1));
      jumpInput.value = total > 0 ? String(currentReviewFrameIndex + 1) : "0";
    }
  }

  function jumpToReviewFrame(rawValue) {
    const reviewFrames = getReviewFrames();
    if (!reviewFrames.length) return;
    const parsed = Number(rawValue);
    if (!Number.isFinite(parsed)) return;
    const index = Math.trunc(parsed) - 1;
    if (index < 0 || index >= reviewFrames.length) return;
    selectReviewFrame(index);
  }

  function drawOverlay() {
    const frame = getCurrentFrame();
    const review = getCurrentReview();
    if (!frame || !frameImage.complete || !review) return;
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
      review_version: "frame_review_v2",
      frame_reviews: getReviewFrames().map((reviewFrame) => {
        const review = ensureReviewEntry(reviewFrame);
        return {
          frame_idx: Number(review.frame_idx),
          confirmed: Boolean(review.confirmed),
          left_mode: review.left_mode,
          right_mode: review.right_mode,
          left_track_id: review.left_track_id == null ? null : Number(review.left_track_id),
          right_track_id: review.right_track_id == null ? null : Number(review.right_track_id),
          left_manual_bbox_xyxy_orig: null,
          right_manual_bbox_xyxy_orig: null,
        };
      }),
    };
  }

  async function submitReview() {
    markReviewConfirmed();
    renderReviewState();
    if (!allConfirmed()) {
      submitMessage.textContent = "还有审核帧未确认，不能提交。";
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
      window.location.href = `/mediapipe/clip/${data.next_clip_id}?start_rank=${startRank}`;
      return;
    }
    submitMessage.textContent = "提交成功，没有更多 ready episode。";
  }

  function hydrateFromExistingReview(reviewPayload) {
    if (!reviewPayload || typeof reviewPayload !== "object") return;
    const reviewVersion = String(reviewPayload.review_version || "");
    if (reviewVersion === "frame_review_v2") {
      (reviewPayload.frame_reviews || []).forEach((item) => {
        const frameIdx = Number(item.frame_idx);
        frameReviewByFrame.set(frameIdx, {
          frame_idx: frameIdx,
          confirmed: Boolean(item.confirmed),
          left_mode: item.left_mode === "manual_box" ? "visible_unrecoverable" : (item.left_mode || "not_visible"),
          right_mode: item.right_mode === "manual_box" ? "visible_unrecoverable" : (item.right_mode || "not_visible"),
          left_track_id: item.left_track_id == null ? null : Number(item.left_track_id),
          right_track_id: item.right_track_id == null ? null : Number(item.right_track_id),
        });
      });
      return;
    }
    if (reviewVersion !== "keyframe_v1") return;
    (reviewPayload.keyframe_reviews || []).forEach((item) => {
      const frameIdx = Number(item.frame_idx);
      frameReviewByFrame.set(frameIdx, {
        frame_idx: frameIdx,
        confirmed: Boolean(item.confirmed),
        left_mode: item.left_track_id != null ? "track" : item.left_missing_box ? "visible_unrecoverable" : "not_visible",
        right_mode: item.right_track_id != null ? "track" : item.right_missing_box ? "visible_unrecoverable" : "not_visible",
        left_track_id: item.left_track_id == null ? null : Number(item.left_track_id),
        right_track_id: item.right_track_id == null ? null : Number(item.right_track_id),
      });
    });
  }

  overlayCanvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });

  overlayCanvas.addEventListener("mousedown", (event) => {
    event.preventDefault();
    const rect = overlayCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (event.button !== 0 && event.button !== 2) return;
    const candidates = drawnBoxes
      .filter((item) => x >= item.left && x <= item.right && y >= item.top && y <= item.bottom)
      .sort((a, b) => a.area - b.area);
    if (!candidates.length) return;
    assignTrack(event.button === 0 ? "left" : "right", candidates[0].track_id);
  });

  leftMissingButton.onclick = () => toggleUnrecoverable("left");
  rightMissingButton.onclick = () => toggleUnrecoverable("right");
  leftNotVisibleButton.onclick = () => setSideNotVisible("left");
  rightNotVisibleButton.onclick = () => setSideNotVisible("right");
  if (jumpGoButton) {
    jumpGoButton.onclick = () => jumpToReviewFrame(jumpInput?.value);
  }
  if (jumpInput) {
    jumpInput.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      jumpToReviewFrame(jumpInput.value);
    });
  }
  prevKeyframeButton.onclick = () => {
    selectReviewFrame(currentReviewFrameIndex - 1);
  };
  nextKeyframeButton.onclick = () => {
    selectReviewFrame(currentReviewFrameIndex + 1);
  };
  submitButton.onclick = submitReview;

  document.addEventListener("keydown", (event) => {
    const target = event.target;
    const tagName = typeof target?.tagName === "string" ? target.tagName.toLowerCase() : "";
    const isEditable = Boolean(target?.isContentEditable) || tagName === "input" || tagName === "textarea" || tagName === "select";
    if (isEditable) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      selectReviewFrame(currentReviewFrameIndex - 1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      selectReviewFrame(currentReviewFrameIndex + 1);
    } else if (event.key === "q" || event.key === "Q") {
      event.preventDefault();
      setSideNotVisible("left");
    } else if (event.key === "e" || event.key === "E") {
      event.preventDefault();
      setSideNotVisible("right");
    } else if (event.key === "a" || event.key === "A") {
      event.preventDefault();
      toggleUnrecoverable("left");
    } else if (event.key === "d" || event.key === "D") {
      event.preventDefault();
      toggleUnrecoverable("right");
    }
  });

  window.addEventListener("resize", () => {
    drawOverlay();
  });

  async function boot() {
    const response = await fetch(`/api/mediapipe/clips/${clipId}`);
    const data = await response.json();
    bundle = data.bundle;
    if (!bundle) {
      titleNode.textContent = "这个 episode 还没有可用的 bundle。";
      return;
    }
    if (!getBaseKeyframes().length) {
      titleNode.textContent = "这个 episode 没有可审核的关键帧。";
      return;
    }
    invalidateReviewFrames();
    (bundle.frame_tracks || []).forEach((item) => {
      frameTracksByFrameIdx.set(Number(item.frame_idx), item);
    });
    (bundle.frames || []).forEach((item) => {
      frameMetaByFrameIdx.set(Number(item.frame_idx), item);
    });
    titleNode.textContent = `${bundle.episode_name} · 全 episode · ${Number(bundle.num_frames || (bundle.frames || []).length)} 帧`;
    subtitleNode.textContent = `${bundle.dirty_reason} · ${getBaseKeyframes().length} 个基础关键帧 · ${(bundle.segments || []).length} 个稳定段`;
    hydrateFromExistingReview(data.clip?.review_payload_json);
    const firstPending = getReviewFrames().findIndex((frame) => !ensureReviewEntry(frame)?.confirmed);
    currentReviewFrameIndex = firstPending >= 0 ? firstPending : 0;
    selectReviewFrame(currentReviewFrameIndex);
  }

  boot().catch((error) => {
    titleNode.textContent = String(error);
  });
}
