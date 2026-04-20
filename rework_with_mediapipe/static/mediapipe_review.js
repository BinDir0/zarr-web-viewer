const dashboardNode = document.getElementById("mpr-dashboard");
const reviewNode = document.getElementById("mpr-review-app");

function renderSummary(summary) {
  const countsNode = document.getElementById("summaryCounts");
  const startLink = document.getElementById("mpr-start-link");
  const startEmpty = document.getElementById("mpr-start-empty");
  const startActions = document.getElementById("mpr-start-actions");
  const startRankInput = document.getElementById("mpr-start-rank");
  const startRank = Math.max(1, Number(summary.requested_start_rank || startRankInput?.value || 1));
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
        startLink.dataset.clipId = String(summary.next_clip_id);
        startLink.dataset.startRank = String(startRank);
        startLink.dataset.loading = "0";
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

async function fetchSummary(startRank = 1, requestSeq = null) {
  const rank = Math.max(1, Number(startRank || 1));
  const response = await fetch(`/api/mediapipe/summary?start_rank=${encodeURIComponent(rank)}`);
  const data = await response.json();
  if (requestSeq !== null && requestSeq !== fetchSummary.latestRequestSeq) return;
  if (response.ok && data.success) {
    renderSummary(data);
  }
}

if (dashboardNode) {
  const startRankInput = document.getElementById("mpr-start-rank");
  const startLink = document.getElementById("mpr-start-link");
  const startEmpty = document.getElementById("mpr-start-empty");
  fetchSummary.latestRequestSeq = 0;
  let refreshSummaryTimer = null;
  const currentStartRank = () => Math.max(1, Number(startRankInput?.value || 1));
  const updateStartRankUrl = (startRank) => {
    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.set("start_rank", String(startRank));
    window.history.replaceState({}, "", nextUrl);
  };
  const setPendingStartLink = (startRank) => {
    if (!startLink) return;
    startLink.href = "#";
    startLink.textContent = `打开第 ${startRank} 条 Episode`;
    startLink.dataset.loading = "0";
    startLink.style.display = "";
    if (startEmpty) startEmpty.style.display = "none";
  };
  const refreshSummary = () => {
    const startRank = Math.max(1, Number(startRankInput?.value || 1));
    if (startRankInput) {
      startRankInput.value = String(startRank);
    }
    updateStartRankUrl(startRank);
    setPendingStartLink(startRank);
    fetchSummary.latestRequestSeq += 1;
    return fetchSummary(startRank, fetchSummary.latestRequestSeq);
  };
  const scheduleSummaryRefresh = () => {
    const startRank = currentStartRank();
    if (startRankInput) startRankInput.value = String(startRank);
    updateStartRankUrl(startRank);
    setPendingStartLink(startRank);
    if (refreshSummaryTimer !== null) {
      window.clearTimeout(refreshSummaryTimer);
    }
    refreshSummaryTimer = window.setTimeout(() => {
      refreshSummaryTimer = null;
      refreshSummary().catch((error) => console.error(error));
    }, 300);
  };
  async function openCurrentStartRank() {
    const startRank = currentStartRank();
    if (refreshSummaryTimer !== null) {
      window.clearTimeout(refreshSummaryTimer);
      refreshSummaryTimer = null;
    }
    if (startLink) {
      startLink.textContent = `打开第 ${startRank} 条 Episode...`;
      startLink.dataset.loading = "1";
    }
    const response = await fetch(`/api/mediapipe/clips/next?start_rank=${encodeURIComponent(startRank)}`);
    const data = await response.json();
    if (!response.ok || !data.success || !data.clip_id) {
      if (startLink) startLink.style.display = "none";
      if (startEmpty) {
        startEmpty.textContent = `从第 ${startRank} 条开始时，当前没有已预处理 episode`;
        startEmpty.style.display = "";
      }
      return;
    }
    window.location.href = `/mediapipe/clip/${data.clip_id}?start_rank=${startRank}`;
  }
  if (startLink) {
    startLink.addEventListener("click", (event) => {
      event.preventDefault();
      openCurrentStartRank().catch((error) => {
        if (startEmpty) {
          startEmpty.textContent = String(error);
          startEmpty.style.display = "";
        }
      });
    });
  }
  if (startRankInput) {
    startRankInput.addEventListener("input", () => {
      scheduleSummaryRefresh();
    });
    startRankInput.addEventListener("change", () => {
      scheduleSummaryRefresh();
    });
  }
  refreshSummary().catch((error) => console.error(error));
}

if (reviewNode) {
  let currentClipId = Number(reviewNode.dataset.clipId);
  let currentStartRank = Math.max(1, Number(reviewNode.dataset.startRank || 1));
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
  const leftAbsentButton = document.getElementById("mpr-left-not-visible");
  const rightAbsentButton = document.getElementById("mpr-right-not-visible");
  const leftUnusableButton = document.getElementById("mpr-left-unusable");
  const rightUnusableButton = document.getElementById("mpr-right-unusable");
  const leftManualButton = document.getElementById("mpr-left-manual-box");
  const rightManualButton = document.getElementById("mpr-right-manual-box");
  const drawStatus = document.getElementById("mpr-draw-status");
  const assignmentSummary = document.getElementById("mpr-assignment-summary");
  const stripSummary = document.getElementById("mpr-strip-summary");
  const episodeRankInput = document.getElementById("mpr-episode-rank-input");
  const episodeRankGoButton = document.getElementById("mpr-episode-rank-go");
  const episodeJumpStatus = document.getElementById("mpr-episode-jump-status");
  const jumpInput = document.getElementById("mpr-jump-input");
  const jumpGoButton = document.getElementById("mpr-jump-go");
  const submitButton = document.getElementById("mpr-submit-review");
  const submitMessage = document.getElementById("mpr-submit-message");

  let bundle = null;
  let currentReviewFrameIndex = 0;
  let drawnBoxes = [];
  let activeDrawSide = null;
  let draftBox = null;
  let dragState = null;
  const frameMetaByFrameIdx = new Map();
  const frameProposalsByFrameIdx = new Map();
  const frameReviewByFrame = new Map();
  const imagePreloadCache = new Map();
  const clipDataCache = new Map();
  let loadSequence = 0;

  async function parseJsonResponse(response) {
    const rawText = await response.text();
    try {
      return JSON.parse(rawText);
    } catch (error) {
      const compact = String(rawText || "").trim().slice(0, 300);
      throw new Error(compact || `HTTP ${response.status}`);
    }
  }

  function fetchClipData(targetClipId) {
    const normalizedClipId = Number(targetClipId);
    if (!clipDataCache.has(normalizedClipId)) {
      const promise = fetch(`/api/mediapipe/clips/${normalizedClipId}`)
        .then(async (response) => {
          const data = await parseJsonResponse(response);
          if (!response.ok || !data.success) {
            throw new Error(data.message || `加载 episode ${normalizedClipId} 失败`);
          }
          return data;
        })
        .catch((error) => {
          clipDataCache.delete(normalizedClipId);
          throw error;
        });
      clipDataCache.set(normalizedClipId, promise);
    }
    return clipDataCache.get(normalizedClipId);
  }

  function getReviewFrames() {
    return bundle?.review_frames || [];
  }

  function ensureReviewEntry(frame) {
    if (!frame) return null;
    const frameIdx = Number(frame.frame_idx);
    let review = frameReviewByFrame.get(frameIdx);
    if (!review) {
      review = {
        frame_idx: frameIdx,
        confirmed: false,
        left_mode: "absent",
        right_mode: "absent",
        left_proposal_id: null,
        right_proposal_id: null,
        left_manual_bbox_xyxy_orig: null,
        right_manual_bbox_xyxy_orig: null,
      };
      frameReviewByFrame.set(frameIdx, review);
    }
    return review;
  }

  function getCurrentFrame() {
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

  function allConfirmed() {
    return getReviewFrames().every((frame) => ensureReviewEntry(frame)?.confirmed);
  }

  function frameAssetUrl(frame) {
    return `/mediapipe/assets/${frame.relpath}`;
  }

  function ensureImagePreloaded(frame) {
    if (!frame?.relpath) return null;
    const src = frameAssetUrl(frame);
    let cached = imagePreloadCache.get(src);
    if (cached) return cached;
    cached = {
      src,
      displaySrc: src,
      img: null,
      objectUrl: null,
      promise: null,
    };
    cached.promise = fetch(src, { cache: "force-cache" })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.blob();
      })
      .then(
        (blob) =>
          new Promise((resolve, reject) => {
            const img = new Image();
            const objectUrl = URL.createObjectURL(blob);
            img.decoding = "async";
            img.loading = "eager";
            img.onload = async () => {
              try {
                if (img.decode) await img.decode();
              } catch (error) {
                // Some browsers reject decode() for already-decoded cached images.
              }
              cached.img = img;
              cached.objectUrl = objectUrl;
              cached.displaySrc = objectUrl;
              resolve(cached);
            };
            img.onerror = reject;
            img.src = objectUrl;
          })
      )
      .catch(
        () =>
          new Promise((resolve, reject) => {
            const img = new Image();
            img.decoding = "async";
            img.loading = "eager";
            img.onload = async () => {
              try {
                if (img.decode) await img.decode();
              } catch (error) {
                // Some browsers reject decode() for already-decoded cached images.
              }
              cached.img = img;
              cached.displaySrc = src;
              resolve(cached);
            };
            img.onerror = reject;
            img.src = src;
          })
      );
    imagePreloadCache.set(src, cached);
    return cached;
  }

  function setFrameImageSource(frame, cached) {
    const src = frameAssetUrl(frame);
    const displaySrc = cached?.displaySrc || src;
    frameImage.dataset.src = src;
    frameImage.onload = () => {
      if (frameImage.dataset.src === src) drawOverlay();
    };
    if (frameImage.src !== displaySrc) {
      frameImage.src = displaySrc;
    } else if (frameImage.complete) {
      drawOverlay();
    }
    cached?.promise
      ?.then((entry) => {
        if (frameImage.dataset.src !== src) return;
        if (entry.displaySrc && frameImage.src !== entry.displaySrc) {
          frameImage.src = entry.displaySrc;
          return;
        }
        try {
          if (frameImage.complete) drawOverlay();
        } catch (error) {
          // drawOverlay is guarded by image completeness.
        }
      })
      .catch(() => {});
  }

  function warmImageWindow(centerIndex, radius = 6) {
    const reviewFrames = getReviewFrames();
    const start = Math.max(0, centerIndex - radius);
    const end = Math.min(reviewFrames.length - 1, centerIndex + radius);
    for (let index = start; index <= end; index += 1) {
      ensureImagePreloaded(reviewFrames[index]);
    }
  }

  function warmReviewImagesProgressively(reviewFrames) {
    let index = 0;
    const tick = () => {
      const batchEnd = Math.min(reviewFrames.length, index + 4);
      while (index < batchEnd) {
        ensureImagePreloaded(reviewFrames[index]);
        index += 1;
      }
      if (index < reviewFrames.length) {
        window.setTimeout(tick, 80);
      }
    };
    window.setTimeout(tick, 120);
  }

  function warmBundleImages(compactBundle, maxFrames = 10) {
    (compactBundle?.review_frames || []).slice(0, maxFrames).forEach((frame) => ensureImagePreloaded(frame));
  }

  function renderCurrentFrameImage() {
    const frame = getCurrentFrame();
    if (!frame) return;
    const src = frameAssetUrl(frame);
    const cached = ensureImagePreloaded(frame);
    warmImageWindow(currentReviewFrameIndex);
    if (frameImage.dataset.src === src && frameImage.complete) {
      drawOverlay();
      return;
    }
    setFrameImageSource(frame, cached);
  }

  function getProposalList(frameIdx) {
    return frameProposalsByFrameIdx.get(Number(frameIdx))?.proposals || [];
  }

  function roleForProposal(proposalId) {
    const review = getCurrentReview();
    if (!review) return "neutral";
    if (review.left_mode === "proposal" && review.left_proposal_id === proposalId) return "left";
    if (review.right_mode === "proposal" && review.right_proposal_id === proposalId) return "right";
    return "neutral";
  }

  function sideSummary(review, side) {
    const prefix = side === "left" ? "左手" : "右手";
    const mode = review?.[`${side}_mode`] || "absent";
    if (mode === "proposal") return `${prefix}=候选框`;
    if (mode === "manual_box") return `${prefix}=手动画框`;
    if (mode === "unusable") return `${prefix}=不可用`;
    return `${prefix}=不可见`;
  }

  function frameRoleSummary(review) {
    if (!review) return "未初始化";
    const confirmLabel = review.confirmed ? "已确认" : "待确认";
    return `${sideSummary(review, "left")} · ${sideSummary(review, "right")} · ${confirmLabel}`;
  }

  function clearSide(review, side, mode) {
    review[`${side}_mode`] = mode;
    review[`${side}_proposal_id`] = null;
    review[`${side}_manual_bbox_xyxy_orig`] = null;
    review.confirmed = true;
  }

  function setSideAbsent(side) {
    const review = getCurrentReview();
    if (!review) return;
    clearSide(review, side, "absent");
    activeDrawSide = null;
    renderReviewState();
    drawOverlay();
  }

  function setSideUnusable(side) {
    const review = getCurrentReview();
    if (!review) return;
    clearSide(review, side, "unusable");
    activeDrawSide = null;
    renderReviewState();
    drawOverlay();
  }

  function beginManualBox(side) {
    activeDrawSide = side;
    draftBox = null;
    dragState = null;
    renderReviewState();
    drawOverlay();
  }

  function assignProposal(side, proposalId) {
    const review = getCurrentReview();
    if (!review) return;
    if (review[`${side}_mode`] === "proposal" && review[`${side}_proposal_id`] === proposalId) {
      clearSide(review, side, "absent");
    } else {
      review[`${side}_mode`] = "proposal";
      review[`${side}_proposal_id`] = proposalId;
      review[`${side}_manual_bbox_xyxy_orig`] = null;
      review.confirmed = true;
    }
    const otherSide = side === "left" ? "right" : "left";
    if (review[`${otherSide}_mode`] === "proposal" && review[`${otherSide}_proposal_id`] === proposalId) {
      clearSide(review, otherSide, "absent");
    }
    activeDrawSide = null;
    renderReviewState();
    drawOverlay();
  }

  function saveManualBox(side, bboxOrig) {
    const review = getCurrentReview();
    if (!review) return;
    review[`${side}_mode`] = "manual_box";
    review[`${side}_proposal_id`] = null;
    review[`${side}_manual_bbox_xyxy_orig`] = bboxOrig;
    review.confirmed = true;
    activeDrawSide = null;
    draftBox = null;
    dragState = null;
    renderReviewState();
    drawOverlay();
  }

  function selectReviewFrame(index) {
    const reviewFrames = getReviewFrames();
    if (!reviewFrames.length) return;
    const currentFrame = getCurrentFrame();
    if (currentFrame) {
      markReviewConfirmed(currentFrame);
    }
    currentReviewFrameIndex = Math.max(0, Math.min(index, reviewFrames.length - 1));
    activeDrawSide = null;
    draftBox = null;
    dragState = null;
    renderCurrentFrameImage();
    renderReviewState();
  }

  function renderReviewFrameSummary() {
    const reviewFrames = getReviewFrames();
    const total = reviewFrames.length;
    const confirmed = reviewFrames.filter((frame) => ensureReviewEntry(frame)?.confirmed).length;
    stripSummary.textContent = `${confirmed}/${total} 已确认`;
    if (jumpInput) {
      jumpInput.min = total > 0 ? "1" : "0";
      jumpInput.max = String(Math.max(1, total));
      jumpInput.value = total > 0 ? String(currentReviewFrameIndex + 1) : "0";
    }
  }

  function renderReviewState() {
    const reviewFrame = getCurrentFrame();
    const review = getCurrentReview();
    if (!reviewFrame || !review) return;
    const reviewFrames = getReviewFrames();
    const reasons = (reviewFrame.reasons || []).join(" · ") || "无特殊原因";
    keyframeLabel.textContent = `审核帧 ${currentReviewFrameIndex + 1}/${reviewFrames.length} · frame ${reviewFrame.frame_idx}`;
    keyframeReasons.textContent = `原因: ${reasons} · uncertainty ${Number(reviewFrame.uncertainty || 0).toFixed(2)}`;
    keyframeStatus.textContent = review.confirmed ? "已确认" : "待确认";
    leftAbsentButton.classList.toggle("is-active", review.left_mode === "absent");
    rightAbsentButton.classList.toggle("is-active", review.right_mode === "absent");
    leftUnusableButton.classList.toggle("is-active", review.left_mode === "unusable");
    rightUnusableButton.classList.toggle("is-active", review.right_mode === "unusable");
    leftManualButton.classList.toggle("is-active", activeDrawSide === "left");
    rightManualButton.classList.toggle("is-active", activeDrawSide === "right");
    assignmentSummary.textContent = frameRoleSummary(review);
    drawStatus.textContent = activeDrawSide ? `正在为${activeDrawSide === "left" ? "左手" : "右手"}拖拽画框。按 Esc 取消。` : "当前未启用手动画框。";
    viewerHint.textContent = "左键点框=左手，右键点框=右手；1/2 启用手动画框；Q/E=不可见；A/D=不可用；←/→ 切换。";
    submitButton.disabled = !allConfirmed();
    prevKeyframeButton.disabled = currentReviewFrameIndex <= 0;
    nextKeyframeButton.disabled = currentReviewFrameIndex >= reviewFrames.length - 1;
    renderReviewFrameSummary();
  }

  function drawLabel(ctx, x, y, text, color) {
    ctx.font = "bold 14px sans-serif";
    const textWidth = ctx.measureText(text).width;
    const labelX = x;
    const labelY = Math.max(22, y);
    ctx.fillStyle = color;
    ctx.fillRect(labelX, labelY - 18, textWidth + 14, 22);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, labelX + 7, labelY - 4);
  }

  function drawOverlay() {
    const frame = getCurrentFrame();
    const review = getCurrentReview();
    if (!frame || !review || !frameImage.complete) return;
    overlayCanvas.width = frameImage.clientWidth;
    overlayCanvas.height = frameImage.clientHeight;
    const ctx = overlayCanvas.getContext("2d");
    const naturalWidth = frameImage.naturalWidth || 1;
    const naturalHeight = frameImage.naturalHeight || 1;
    const sx = overlayCanvas.width / naturalWidth;
    const sy = overlayCanvas.height / naturalHeight;
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    drawnBoxes = [];

    getProposalList(frame.frame_idx).forEach((proposal, proposalIndex) => {
      const proposalId = String(proposal.proposal_id);
      const role = roleForProposal(proposalId);
      const color = role === "left" ? "#0f766e" : role === "right" ? "#b91c1c" : "#2563eb";
      const [x1, y1, x2, y2] = proposal.bbox_xyxy;
      const left = x1 * sx;
      const top = y1 * sy;
      const width = (x2 - x1) * sx;
      const height = (y2 - y1) * sy;
      ctx.strokeStyle = color;
      ctx.lineWidth = role === "neutral" ? 3 : 5;
      ctx.strokeRect(left, top, width, height);
      const detector = proposal.detector_name === "mediapipe_image" ? "MP" : "YOLO";
      const labelPrefix = role === "left" ? "L" : role === "right" ? "R" : `P${proposalIndex + 1}`;
      drawLabel(ctx, left, top, `${labelPrefix} · ${detector} ${Number(proposal.score || 0).toFixed(2)}`, color);
      drawnBoxes.push({
        proposal_id: proposalId,
        left,
        top,
        right: left + width,
        bottom: top + height,
        area: width * height,
      });
    });

    ["left", "right"].forEach((side) => {
      if (review[`${side}_mode`] !== "manual_box" || !review[`${side}_manual_bbox_xyxy_orig`]) return;
      const frameMeta = frameMetaByFrameIdx.get(Number(frame.frame_idx));
      if (!frameMeta) return;
      const [origWidth, origHeight] = frameMeta.orig_size || [1, 1];
      const [x1, y1, x2, y2] = review[`${side}_manual_bbox_xyxy_orig`];
      const left = (x1 / Math.max(origWidth, 1)) * overlayCanvas.width;
      const top = (y1 / Math.max(origHeight, 1)) * overlayCanvas.height;
      const width = ((x2 - x1) / Math.max(origWidth, 1)) * overlayCanvas.width;
      const height = ((y2 - y1) / Math.max(origHeight, 1)) * overlayCanvas.height;
      const color = side === "left" ? "#ca8a04" : "#7c3aed";
      ctx.strokeStyle = color;
      ctx.setLineDash([8, 6]);
      ctx.lineWidth = 4;
      ctx.strokeRect(left, top, width, height);
      ctx.setLineDash([]);
      drawLabel(ctx, left, top, `${side === "left" ? "L" : "R"} · manual`, color);
    });

    if (draftBox) {
      const color = activeDrawSide === "left" ? "#ca8a04" : "#7c3aed";
      ctx.strokeStyle = color;
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 3;
      ctx.strokeRect(draftBox.left, draftBox.top, draftBox.width, draftBox.height);
      ctx.setLineDash([]);
    }
  }

  function buildPayload() {
    return {
      review_version: "frame_review_v3",
      frame_reviews: getReviewFrames().map((frame) => {
        const review = ensureReviewEntry(frame);
        return {
          frame_idx: Number(review.frame_idx),
          confirmed: Boolean(review.confirmed),
          left_mode: review.left_mode,
          right_mode: review.right_mode,
          left_proposal_id: review.left_proposal_id,
          right_proposal_id: review.right_proposal_id,
          left_manual_bbox_xyxy_orig: review.left_manual_bbox_xyxy_orig,
          right_manual_bbox_xyxy_orig: review.right_manual_bbox_xyxy_orig,
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
    const response = await fetch(`/api/mediapipe/clips/${currentClipId}/submit-review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    const data = await parseJsonResponse(response);
    if (!response.ok || !data.success) {
      submitMessage.textContent = data.message || "提交失败";
      return;
    }
    if (data.next_clip_id) {
      loadClip(data.next_clip_id, { pushUrl: true, startRankValue: currentStartRank }).catch((error) => {
        submitMessage.textContent = String(error);
      });
      return;
    }
    submitMessage.textContent = "提交成功，没有更多 ready episode。";
  }

  function hydrateFromExistingReview(reviewPayload) {
    if (!reviewPayload || typeof reviewPayload !== "object") return;
    if (String(reviewPayload.review_version || "") !== "frame_review_v3") return;
    (reviewPayload.frame_reviews || []).forEach((item) => {
      const frameIdx = Number(item.frame_idx);
      frameReviewByFrame.set(frameIdx, {
        frame_idx: frameIdx,
        confirmed: Boolean(item.confirmed),
        left_mode: item.left_mode || "absent",
        right_mode: item.right_mode || "absent",
        left_proposal_id: item.left_proposal_id || null,
        right_proposal_id: item.right_proposal_id || null,
        left_manual_bbox_xyxy_orig: item.left_manual_bbox_xyxy_orig || null,
        right_manual_bbox_xyxy_orig: item.right_manual_bbox_xyxy_orig || null,
      });
    });
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

  function resetClipState() {
    bundle = null;
    currentReviewFrameIndex = 0;
    drawnBoxes = [];
    activeDrawSide = null;
    draftBox = null;
    dragState = null;
    frameMetaByFrameIdx.clear();
    frameProposalsByFrameIdx.clear();
    frameReviewByFrame.clear();
    submitMessage.textContent = "";
    assignmentSummary.textContent = "";
    keyframeLabel.textContent = "";
    keyframeReasons.textContent = "";
    keyframeStatus.textContent = "";
    stripSummary.textContent = "";
    frameImage.removeAttribute("src");
    frameImage.dataset.src = "";
    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  async function preloadNextEpisode(afterClipId) {
    try {
      const response = await fetch(`/api/mediapipe/clips/next?after_clip_id=${encodeURIComponent(afterClipId)}`);
      const data = await parseJsonResponse(response);
      if (!response.ok || !data.success || !data.clip_id || Number(data.clip_id) === Number(currentClipId)) return;
      const nextData = await fetchClipData(Number(data.clip_id));
      warmBundleImages(nextData.bundle, 10);
    } catch (error) {
      console.debug("next episode preload failed", error);
    }
  }

  async function loadClip(targetClipId, options = {}) {
    const normalizedClipId = Number(targetClipId);
    const nextStartRank = Math.max(1, Number(options.startRankValue || currentStartRank || 1));
    const sequence = (loadSequence += 1);
    titleNode.textContent = `Loading episode ${normalizedClipId}...`;
    subtitleNode.textContent = "";
    submitButton.disabled = true;
    let data;
    try {
      data = await fetchClipData(normalizedClipId);
    } catch (error) {
      if (sequence === loadSequence) titleNode.textContent = String(error);
      throw error;
    }
    if (sequence !== loadSequence) return;

    currentClipId = normalizedClipId;
    currentStartRank = nextStartRank;
    reviewNode.dataset.clipId = String(currentClipId);
    reviewNode.dataset.startRank = String(currentStartRank);
    if (episodeRankInput) episodeRankInput.value = String(currentStartRank);
    if (options.pushUrl) {
      const nextUrl = `/mediapipe/clip/${currentClipId}?start_rank=${currentStartRank}`;
      window.history.pushState({ clipId: currentClipId, startRank: currentStartRank }, "", nextUrl);
    }

    resetClipState();
    bundle = data.bundle;
    if (!bundle) {
      titleNode.textContent = "这个 episode 还没有可用的 bundle。";
      return;
    }
    (bundle.frames || []).forEach((item) => {
      frameMetaByFrameIdx.set(Number(item.frame_idx), item);
    });
    (bundle.frame_proposals || []).forEach((item) => {
      frameProposalsByFrameIdx.set(Number(item.frame_idx), item);
    });
    titleNode.textContent = `${bundle.episode_name} · 全 episode · ${Number(bundle.num_frames || 0)} 帧`;
    subtitleNode.textContent = `${bundle.dirty_reason} · ${getReviewFrames().length} 个审核帧 · clip ${currentClipId}`;
    hydrateFromExistingReview(data.clip?.review_payload_json);
    const firstPending = getReviewFrames().findIndex((frame) => !ensureReviewEntry(frame)?.confirmed);
    currentReviewFrameIndex = firstPending >= 0 ? firstPending : 0;
    selectReviewFrame(currentReviewFrameIndex);
    warmReviewImagesProgressively(getReviewFrames());
    preloadNextEpisode(currentClipId);
  }

  async function jumpToEpisodeRank(rawValue) {
    const parsed = Number(rawValue);
    if (!Number.isFinite(parsed)) return;
    const rank = Math.max(1, Math.trunc(parsed));
    if (episodeRankInput) episodeRankInput.value = String(rank);
    if (episodeJumpStatus) episodeJumpStatus.textContent = `正在查找第 ${rank} 条 episode...`;
    const response = await fetch(`/api/mediapipe/clips/next?start_rank=${encodeURIComponent(rank)}`);
    const data = await parseJsonResponse(response);
    if (!response.ok || !data.success) {
      if (episodeJumpStatus) episodeJumpStatus.textContent = data.message || "跳转失败";
      return;
    }
    if (!data.clip_id) {
      if (episodeJumpStatus) episodeJumpStatus.textContent = `当前没有第 ${rank} 条已预处理 episode。`;
      return;
    }
    if (episodeJumpStatus) episodeJumpStatus.textContent = `正在打开第 ${rank} 条 episode: clip ${data.clip_id}`;
    await loadClip(Number(data.clip_id), { pushUrl: true, startRankValue: rank });
    if (episodeJumpStatus) episodeJumpStatus.textContent = `已打开第 ${rank} 条 episode: clip ${data.clip_id}`;
  }

  function canvasPointToOrig(event) {
    const frame = getCurrentFrame();
    const frameMeta = frameMetaByFrameIdx.get(Number(frame?.frame_idx));
    if (!frameMeta) return null;
    const rect = overlayCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const [origWidth, origHeight] = frameMeta.orig_size || [1, 1];
    return {
      canvasX: x,
      canvasY: y,
      origX: (x / Math.max(overlayCanvas.width, 1)) * origWidth,
      origY: (y / Math.max(overlayCanvas.height, 1)) * origHeight,
    };
  }

  overlayCanvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });

  overlayCanvas.addEventListener("mousedown", (event) => {
    event.preventDefault();
    if (activeDrawSide) {
      if (event.button !== 0) return;
      const point = canvasPointToOrig(event);
      if (!point) return;
      dragState = {
        side: activeDrawSide,
        startOrigX: point.origX,
        startOrigY: point.origY,
        startCanvasX: point.canvasX,
        startCanvasY: point.canvasY,
      };
      draftBox = { left: point.canvasX, top: point.canvasY, width: 0, height: 0 };
      drawOverlay();
      return;
    }
    if (event.button !== 0 && event.button !== 2) return;
    const rect = overlayCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const candidates = drawnBoxes
      .filter((item) => x >= item.left && x <= item.right && y >= item.top && y <= item.bottom)
      .sort((a, b) => a.area - b.area);
    if (!candidates.length) return;
    assignProposal(event.button === 0 ? "left" : "right", candidates[0].proposal_id);
  });

  overlayCanvas.addEventListener("mousemove", (event) => {
    if (!dragState) return;
    const point = canvasPointToOrig(event);
    if (!point) return;
    draftBox = {
      left: Math.min(dragState.startCanvasX, point.canvasX),
      top: Math.min(dragState.startCanvasY, point.canvasY),
      width: Math.abs(point.canvasX - dragState.startCanvasX),
      height: Math.abs(point.canvasY - dragState.startCanvasY),
    };
    drawOverlay();
  });

  overlayCanvas.addEventListener("mouseup", (event) => {
    if (!dragState) return;
    const point = canvasPointToOrig(event);
    if (!point) return;
    const x1 = Math.min(dragState.startOrigX, point.origX);
    const y1 = Math.min(dragState.startOrigY, point.origY);
    const x2 = Math.max(dragState.startOrigX, point.origX);
    const y2 = Math.max(dragState.startOrigY, point.origY);
    const minSpan = 4;
    const side = dragState.side;
    dragState = null;
    if (x2 - x1 < minSpan || y2 - y1 < minSpan) {
      draftBox = null;
      activeDrawSide = null;
      renderReviewState();
      drawOverlay();
      return;
    }
    saveManualBox(side, [x1, y1, x2, y2]);
  });

  leftAbsentButton.onclick = () => setSideAbsent("left");
  rightAbsentButton.onclick = () => setSideAbsent("right");
  leftUnusableButton.onclick = () => setSideUnusable("left");
  rightUnusableButton.onclick = () => setSideUnusable("right");
  leftManualButton.onclick = () => beginManualBox("left");
  rightManualButton.onclick = () => beginManualBox("right");
  if (episodeRankGoButton) {
    episodeRankGoButton.onclick = () => {
      jumpToEpisodeRank(episodeRankInput?.value).catch((error) => {
        if (episodeJumpStatus) episodeJumpStatus.textContent = String(error);
      });
    };
  }
  if (episodeRankInput) {
    episodeRankInput.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      jumpToEpisodeRank(episodeRankInput.value).catch((error) => {
        if (episodeJumpStatus) episodeJumpStatus.textContent = String(error);
      });
    });
  }
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
  prevKeyframeButton.onclick = () => selectReviewFrame(currentReviewFrameIndex - 1);
  nextKeyframeButton.onclick = () => selectReviewFrame(currentReviewFrameIndex + 1);
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
      setSideAbsent("left");
    } else if (event.key === "e" || event.key === "E") {
      event.preventDefault();
      setSideAbsent("right");
    } else if (event.key === "a" || event.key === "A") {
      event.preventDefault();
      setSideUnusable("left");
    } else if (event.key === "d" || event.key === "D") {
      event.preventDefault();
      setSideUnusable("right");
    } else if (event.key === "1") {
      event.preventDefault();
      beginManualBox("left");
    } else if (event.key === "2") {
      event.preventDefault();
      beginManualBox("right");
    } else if (event.key === "Escape") {
      event.preventDefault();
      activeDrawSide = null;
      draftBox = null;
      dragState = null;
      renderReviewState();
      drawOverlay();
    }
  });

  window.addEventListener("resize", () => drawOverlay());
  window.addEventListener("popstate", () => {
    const match = window.location.pathname.match(/\/mediapipe\/clip\/(\d+)$/);
    if (!match) return;
    const nextRank = Math.max(1, Number(new URL(window.location.href).searchParams.get("start_rank") || currentStartRank || 1));
    loadClip(Number(match[1]), { pushUrl: false, startRankValue: nextRank }).catch((error) => {
      titleNode.textContent = String(error);
    });
  });

  async function boot() {
    await loadClip(currentClipId, { pushUrl: false, startRankValue: currentStartRank });
  }

  boot().catch((error) => {
    titleNode.textContent = String(error);
  });
}
