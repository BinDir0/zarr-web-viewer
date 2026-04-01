let queueMode = 'episode';
let queueItems = [];
let currentOffset = 0;
let totalItemCount = 0;
let sessionId = null;

const FRAME_STATE_SEP = '\u0001';
const LABELS = {
    ok: '',
    missing_annotation: 'A · 双手清晰可见，但缺少标注',
    non_visible_wrong_annotation: 'B · 非双手可见且框标错',
    visible_box_wrong_annotation: 'C · 双手可见但框标错'
};

const frameStates = new Map();
let preloadedBatch = null;
let isPreloading = false;
let preloadAbortController = null;

function getUserId() {
    let userId = localStorage.getItem('userId');
    if (!userId) {
        userId = 'user_' + Date.now() + '_' + Math.random().toString(36).slice(2, 11);
        localStorage.setItem('userId', userId);
    }
    return userId;
}

function nextState(state) {
    if (state === 'ok') return 'missing_annotation';
    if (state === 'missing_annotation') return 'non_visible_wrong_annotation';
    if (state === 'non_visible_wrong_annotation') return 'visible_box_wrong_annotation';
    return 'ok';
}

function frameStateKey(episodeId, frameIndex) {
    return episodeId + FRAME_STATE_SEP + String(frameIndex);
}

function getState(key) {
    return frameStates.get(key) || 'ok';
}

function setState(key, state) {
    if (state === 'ok') {
        frameStates.delete(key);
    } else {
        frameStates.set(key, state);
    }
}

function applyStateToElement(element, state) {
    element.classList.remove(
        'state-ok',
        'state-missing_annotation',
        'state-non_visible_wrong_annotation',
        'state-visible_box_wrong_annotation'
    );
    element.classList.add('state-' + state);
    element.dataset.frameState = state;
    const label = element.querySelector('.frame-mask-label');
    if (label) {
        label.textContent = LABELS[state] || '';
    }
}

function applyReviewedBorder(cardElement, reviewed, state) {
    if (!cardElement) return;
    cardElement.classList.remove('annotated', 'alright', 'bad');
    if (!reviewed) return;
    cardElement.classList.add('annotated');
    cardElement.classList.add(state === 'ok' ? 'alright' : 'bad');
}

function initialEpisodeFrameState(annotation, relativeFrameIndex) {
    if (!annotation || !annotation.has_annotation) {
        return 'ok';
    }
    const reasoned = annotation.reasoned_annotation || null;
    const missingAnnotation = reasoned && Array.isArray(reasoned.missing_annotation)
        ? reasoned.missing_annotation
        : (reasoned && Array.isArray(reasoned.not_clear) ? reasoned.not_clear : []);
    const nonVisibleWrongAnnotation = reasoned && Array.isArray(reasoned.non_visible_wrong_annotation)
        ? reasoned.non_visible_wrong_annotation
        : [];
    const visibleBoxWrongAnnotation = reasoned && Array.isArray(reasoned.visible_box_wrong_annotation)
        ? reasoned.visible_box_wrong_annotation
        : (reasoned && Array.isArray(reasoned.wrong_annotation) ? reasoned.wrong_annotation : (reasoned && Array.isArray(reasoned.bad_box) ? reasoned.bad_box : []));
    if (missingAnnotation.includes(relativeFrameIndex)) return 'missing_annotation';
    if (nonVisibleWrongAnnotation.includes(relativeFrameIndex)) return 'non_visible_wrong_annotation';
    if (visibleBoxWrongAnnotation.includes(relativeFrameIndex)) return 'visible_box_wrong_annotation';
    if (Array.isArray(annotation.bad_frames) && annotation.bad_frames.includes(relativeFrameIndex)) {
        return 'visible_box_wrong_annotation';
    }
    return 'ok';
}

function updateQueueModeMeta(mode, totalItems, itemLabel) {
    queueMode = mode || 'episode';
    totalItemCount = Number.isFinite(totalItems) ? totalItems : 0;
    const label = itemLabel || (queueMode === 'frame' ? 'frames' : 'episodes');

    const startOffsetLabel = document.getElementById('startOffsetLabel');
    const totalInfo = document.getElementById('totalEpisodesInfo');
    const input = document.getElementById('startOffset');
    const container = document.getElementById('episodesContainer');

    if (startOffsetLabel) {
        startOffsetLabel.textContent = queueMode === 'frame' ? '起始 frame：' : '起始 episode：';
    }
    if (input) {
        input.placeholder = queueMode === 'frame'
            ? '输入起始 frame 编号（例如: 0, 210, 420）'
            : '输入起始 episode 编号（例如: 0, 1000, 5000）';
        input.max = Math.max(0, totalItemCount - 1);
    }
    if (totalInfo) {
        if (totalItemCount > 0) {
            totalInfo.textContent = `（总共: ${totalItemCount} 个${label}）`;
            totalInfo.style.color = '#4CAF50';
        } else {
            totalInfo.textContent = '（总数: 加载中...）';
            totalInfo.style.color = '#888';
        }
    }
    if (container) {
        container.classList.toggle('frame-mode', queueMode === 'frame');
    }
}

function updateStats() {
    let totalFrames = 0;
    let missingAnnotation = 0;
    let nonVisibleWrongAnnotation = 0;
    let visibleBoxWrongAnnotation = 0;

    if (queueMode === 'frame') {
        queueItems.forEach((item) => {
            totalFrames += 1;
            const state = getState(frameStateKey(item.episode_id, item.frame_index));
            if (state === 'missing_annotation') missingAnnotation += 1;
            else if (state === 'non_visible_wrong_annotation') nonVisibleWrongAnnotation += 1;
            else if (state === 'visible_box_wrong_annotation') visibleBoxWrongAnnotation += 1;
        });
    } else {
        queueItems.forEach((episode) => {
            if (!Array.isArray(episode.frame_indices)) {
                return;
            }
            totalFrames += episode.frame_indices.length;
            episode.frame_indices.forEach((absoluteFrameIndex) => {
                const relativeFrameIndex = absoluteFrameIndex - (episode.start_idx || 0);
                const state = getState(frameStateKey(episode.id, relativeFrameIndex));
                if (state === 'missing_annotation') missingAnnotation += 1;
                else if (state === 'non_visible_wrong_annotation') nonVisibleWrongAnnotation += 1;
                else if (state === 'visible_box_wrong_annotation') visibleBoxWrongAnnotation += 1;
            });
        });
    }

    const okCount = Math.max(
        0,
        totalFrames - missingAnnotation - nonVisibleWrongAnnotation - visibleBoxWrongAnnotation
    );
    const label = queueMode === 'frame' ? 'frames' : 'episodes';
    const loadedCount = document.getElementById('loadedCount');
    if (loadedCount) {
        loadedCount.textContent = `已加载 ${queueItems.length} / ${totalItemCount || '?'} 个${label} | 正确 ${okCount} | A ${missingAnnotation} | B ${nonVisibleWrongAnnotation} | C ${visibleBoxWrongAnnotation}`;
    }
}

async function cancelPreload() {
    if (isPreloading && preloadAbortController) {
        preloadAbortController.abort();
    }
    preloadedBatch = null;
    const preloadStatus = document.getElementById('preloadStatus');
    if (preloadStatus) preloadStatus.textContent = '';
}

async function preloadNextBatch() {
    if (isPreloading || currentOffset === 0) {
        return;
    }

    isPreloading = true;
    preloadAbortController = new AbortController();
    const preloadStatus = document.getElementById('preloadStatus');
    if (preloadStatus) preloadStatus.textContent = '正在预加载下一批...';

    try {
        const response = await fetch(
            `/api/episodes/sequential?user_id=${getUserId()}&limit=210&offset=${currentOffset}`,
            { signal: preloadAbortController.signal }
        );
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.message || '预加载失败');
        }
        preloadedBatch = { offset: currentOffset, data };
        if (preloadStatus) {
            preloadStatus.textContent = `下一批已预加载 (${data.collected_count || 0})`;
        }
    } catch (error) {
        if (error.name !== 'AbortError') {
            console.warn('预加载失败:', error);
        }
        preloadedBatch = null;
        if (preloadStatus) preloadStatus.textContent = '';
    } finally {
        isPreloading = false;
        preloadAbortController = null;
    }
}

function renderFrameCard(item) {
    const container = document.getElementById('episodesContainer');

    const card = document.createElement('div');
    card.className = 'frame-card';
    card.dataset.frameId = item.frame_id;

    const header = document.createElement('div');
    header.className = 'frame-card-header';
    header.innerHTML = `
        <div class="frame-card-title">
            <span class="dataset-badge">${item.dataset_name || 'Unknown'}</span>
            <span>${item.episode_name || item.episode_id}</span>
        </div>
        <div class="frame-card-meta">#${item.queue_index ?? '?'}</div>
    `;

    const wrapper = document.createElement('div');
    wrapper.className = 'frame-wrapper';
    wrapper.dataset.episodeId = item.episode_id;
    wrapper.dataset.frameIndex = String(item.frame_index);

    const frameIndexLabel = document.createElement('div');
    frameIndexLabel.className = 'frame-index';
    frameIndexLabel.textContent = `Frame ${item.frame_index}`;

    const img = document.createElement('img');
    img.src = item.image;
    img.alt = `${item.episode_id}:${item.frame_index}`;

    const mask = document.createElement('div');
    mask.className = 'frame-mask';
    const maskLabel = document.createElement('div');
    maskLabel.className = 'frame-mask-label';
    mask.appendChild(maskLabel);

    wrapper.appendChild(frameIndexLabel);
    wrapper.appendChild(img);
    wrapper.appendChild(mask);

    const footer = document.createElement('div');
    footer.className = 'frame-card-footer';
    const prob = item.prob_is_bad;
    footer.textContent = `episode_index=${item.episode_index ?? '?'} | prob_is_bad=${typeof prob === 'number' ? prob.toFixed(4) : 'N/A'}`;

    const key = frameStateKey(item.episode_id, item.frame_index);
    const initialState = item.annotation_state || 'ok';
    setState(key, initialState);
    applyStateToElement(wrapper, initialState);
    applyReviewedBorder(card, Boolean(item.was_reviewed), initialState);

    wrapper.onclick = () => {
        const next = nextState(getState(key));
        setState(key, next);
        applyStateToElement(wrapper, next);
        applyReviewedBorder(card, true, next);
        updateStats();
    };

    card.appendChild(header);
    card.appendChild(wrapper);
    card.appendChild(footer);
    container.appendChild(card);
}

function renderEpisodeCard(episode) {
    const container = document.getElementById('episodesContainer');
    const data = episode.raw;

    if (data.stage1_pending) {
        const episodeBlock = document.createElement('div');
        episodeBlock.className = 'episode-block stage1-pending';
        episodeBlock.innerHTML = `
            <div class="episode-header">
                <div class="episode-title">
                    <span class="dataset-badge">${episode.dataset || 'Unknown'}</span>
                    <span>${episode.name || episode.id}</span>
                </div>
                <div class="episode-info">${data.num_frames || '?'} frames</div>
            </div>
            <div class="stage1-pending-label">Stage 1 (detect_track) 尚未完成，请先跳过标注</div>
        `;
        container.appendChild(episodeBlock);
        return;
    }

    if (!Array.isArray(data.images) || data.images.length === 0 || !Array.isArray(data.frame_indices)) {
        return;
    }

    const episodeBlock = document.createElement('div');
    episodeBlock.className = 'episode-block';

    if (data.annotation && data.annotation.has_annotation) {
        const markType = data.annotation.mark_type;
        episodeBlock.classList.add('annotated', markType);
        const statusLabel = document.createElement('div');
        statusLabel.className = `annotation-status ${markType}`;
        if (markType === 'alright') {
            statusLabel.textContent = '已标注：Alright';
        } else {
            const reasoned = data.annotation.reasoned_annotation || {
                missing_annotation: [],
                non_visible_wrong_annotation: [],
                visible_box_wrong_annotation: []
            };
            const missingCount = Array.isArray(reasoned.missing_annotation)
                ? reasoned.missing_annotation.length
                : (Array.isArray(reasoned.not_clear) ? reasoned.not_clear.length : 0);
            const nonVisibleWrongCount = Array.isArray(reasoned.non_visible_wrong_annotation)
                ? reasoned.non_visible_wrong_annotation.length
                : 0;
            const visibleBoxWrongCount = Array.isArray(reasoned.visible_box_wrong_annotation)
                ? reasoned.visible_box_wrong_annotation.length
                : (Array.isArray(reasoned.wrong_annotation) ? reasoned.wrong_annotation.length : (Array.isArray(reasoned.bad_box) ? reasoned.bad_box.length : 0));
            statusLabel.textContent = `已标注：问题帧 (A ${missingCount} / B ${nonVisibleWrongCount} / C ${visibleBoxWrongCount})`;
        }
        episodeBlock.appendChild(statusLabel);
    }

    const header = document.createElement('div');
    header.className = 'episode-header';
    header.innerHTML = `
        <div class="episode-title">
            <span class="dataset-badge">${episode.dataset || 'Unknown'}</span>
            <span>${episode.name || episode.id}</span>
        </div>
        <div class="episode-info">Episode #${episode.episode_index ?? '?'} | ${data.num_frames || '?'} frames</div>
    `;

    const framesContainer = document.createElement('div');
    framesContainer.className = 'frames-container';

    data.images.forEach((imgSrc, idx) => {
        const absoluteFrameIndex = data.frame_indices[idx] || idx;
        const relativeFrameIndex = absoluteFrameIndex - (episode.start_idx || 0);
        const key = frameStateKey(episode.id, relativeFrameIndex);

        const wrapper = document.createElement('div');
        wrapper.className = 'frame-wrapper';

        const frameIndexLabel = document.createElement('div');
        frameIndexLabel.className = 'frame-index';
        frameIndexLabel.textContent = `Frame ${relativeFrameIndex}`;

        const img = document.createElement('img');
        img.src = imgSrc;
        img.alt = `${episode.id}:${relativeFrameIndex}`;

        const mask = document.createElement('div');
        mask.className = 'frame-mask';
        const maskLabel = document.createElement('div');
        maskLabel.className = 'frame-mask-label';
        mask.appendChild(maskLabel);

        wrapper.appendChild(frameIndexLabel);
        wrapper.appendChild(img);
        wrapper.appendChild(mask);

        const initialState = initialEpisodeFrameState(data.annotation, relativeFrameIndex);
        setState(key, initialState);
        applyStateToElement(wrapper, initialState);

        wrapper.onclick = () => {
            const next = nextState(getState(key));
            setState(key, next);
            applyStateToElement(wrapper, next);
            updateStats();
        };

        framesContainer.appendChild(wrapper);
    });

    episodeBlock.appendChild(header);
    episodeBlock.appendChild(framesContainer);
    container.appendChild(episodeBlock);
}

async function loadEpisodes(clearMarks = true) {
    const submitBtn = document.getElementById('submitBtn');
    const submitExitBtn = document.getElementById('submitExitBtn');
    const loading = document.getElementById('loading');
    const container = document.getElementById('episodesContainer');

    if (clearMarks) {
        frameStates.clear();
    }

    submitBtn.disabled = true;
    submitExitBtn.disabled = true;
    loading.style.display = 'block';
    container.innerHTML = '';

    try {
        let data;
        if (preloadedBatch && preloadedBatch.offset === currentOffset) {
            data = preloadedBatch.data;
            preloadedBatch = null;
        } else {
            if (preloadedBatch && preloadedBatch.offset !== currentOffset) {
                await cancelPreload();
            }
            const response = await fetch(`/api/episodes/sequential?user_id=${getUserId()}&limit=210&offset=${currentOffset}`);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
            }
            data = await response.json();
        }

        if (!data.success) {
            throw new Error(data.message || '加载失败');
        }

        updateQueueModeMeta(data.queue_mode, data.total_items ?? data.total_episodes, data.item_label);
        currentOffset = data.next_offset ?? currentOffset;
        localStorage.setItem('sanity_check_offset', String(currentOffset));

        const offsetInput = document.getElementById('startOffset');
        if (offsetInput) {
            offsetInput.value = String(currentOffset);
        }

        if (queueMode === 'frame') {
            queueItems = Array.isArray(data.items) ? data.items : [];
            if (queueItems.length === 0 && data.debug_fail_reasons) {
                alert('本批次 frame 加载失败：\n' + data.debug_fail_reasons.join('\n'));
            }
            for (let i = 0; i < queueItems.length; i += 60) {
                queueItems.slice(i, i + 60).forEach(renderFrameCard);
                await new Promise((resolve) => setTimeout(resolve, 0));
            }
        } else {
            const rawEpisodes = Array.isArray(data.episodes) ? data.episodes : [];
            queueItems = rawEpisodes.map((ep) => ({
                id: ep.episode_id,
                name: ep.episode_name,
                dataset: ep.dataset_name,
                episode_index: ep.episode_index,
                num_frames: ep.num_frames,
                start_idx: ep.start_idx,
                frame_indices: ep.frame_indices,
                raw: ep
            }));
            if (queueItems.length === 0 && data.debug_fail_reasons) {
                alert('本批次 episode 加载失败：\n' + data.debug_fail_reasons.join('\n'));
            }
            for (let i = 0; i < queueItems.length; i += 30) {
                queueItems.slice(i, i + 30).forEach(renderEpisodeCard);
                await new Promise((resolve) => setTimeout(resolve, 0));
            }
        }

        submitBtn.disabled = queueItems.length === 0;
        submitExitBtn.disabled = queueItems.length === 0;
        updateStats();

        if (data.has_more && currentOffset > 0) {
            setTimeout(() => preloadNextBatch(), 80);
        }
    } catch (error) {
        console.error(error);
        alert('加载失败: ' + error.message);
    } finally {
        loading.style.display = 'none';
    }
}

async function submitReviewData() {
    let payload;
    if (queueMode === 'frame') {
        payload = {
            queue_mode: 'frame',
            frames: queueItems.map((item) => ({
                frame_id: item.frame_id,
                episode_id: item.episode_id,
                episode_name: item.episode_name,
                dataset_name: item.dataset_name,
                episode_index: item.episode_index,
                frame_index: item.frame_index,
                state: getState(frameStateKey(item.episode_id, item.frame_index))
            })),
            user_id: getUserId(),
            session_id: sessionId
        };
    } else {
        const episodes = queueItems.map((episode) => {
            const missingAnnotation = [];
            const nonVisibleWrongAnnotation = [];
            const visibleBoxWrongAnnotation = [];
            (episode.frame_indices || []).forEach((absoluteFrameIndex) => {
                const relativeFrameIndex = absoluteFrameIndex - (episode.start_idx || 0);
                const state = getState(frameStateKey(episode.id, relativeFrameIndex));
                if (state === 'missing_annotation') missingAnnotation.push(relativeFrameIndex);
                else if (state === 'non_visible_wrong_annotation') nonVisibleWrongAnnotation.push(relativeFrameIndex);
                else if (state === 'visible_box_wrong_annotation') visibleBoxWrongAnnotation.push(relativeFrameIndex);
            });
            return {
                episode_id: episode.id,
                dataset_name: episode.dataset,
                episode_name: episode.name,
                episode_index: episode.episode_index,
                missing_annotation: missingAnnotation,
                non_visible_wrong_annotation: nonVisibleWrongAnnotation,
                visible_box_wrong_annotation: visibleBoxWrongAnnotation
            };
        });
        payload = {
            queue_mode: 'episode',
            episodes,
            reviewed_episodes: episodes.map((episode) => ({
                episode_id: episode.episode_id,
                dataset: episode.dataset_name,
                episode_name: episode.episode_name,
                episode_index: episode.episode_index,
                has_annotation:
                    episode.missing_annotation.length
                    + episode.non_visible_wrong_annotation.length
                    + episode.visible_box_wrong_annotation.length
                    > 0
            })),
            user_id: getUserId(),
            session_id: sessionId
        };
    }

    const response = await fetch('/api/sanity-check/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!data.success) {
        throw new Error(data.message || '提交失败');
    }
    if (Array.isArray(data.skipped_legacy_conflicts) && data.skipped_legacy_conflicts.length > 0) {
        console.warn('以下 episode 因 legacy 冲突被跳过:', data.skipped_legacy_conflicts);
    }
    return data;
}

async function submitAndLoadNext() {
    const submitBtn = document.getElementById('submitBtn');
    const submitExitBtn = document.getElementById('submitExitBtn');
    submitBtn.disabled = true;
    submitExitBtn.disabled = true;

    try {
        await submitReviewData();
        await loadEpisodes(true);
    } catch (error) {
        console.error(error);
        alert('操作失败: ' + error.message);
        submitBtn.disabled = false;
        submitExitBtn.disabled = false;
    }
}

async function submitAndExit() {
    const submitBtn = document.getElementById('submitBtn');
    const submitExitBtn = document.getElementById('submitExitBtn');
    submitBtn.disabled = true;
    submitExitBtn.disabled = true;

    await cancelPreload();

    try {
        await submitReviewData();
        alert('提交成功。');
        window.close();
        setTimeout(() => {
            document.body.innerHTML = `
                <div style="display:flex;align-items:center;justify-content:center;height:100vh;flex-direction:column;color:#e0e0e0;">
                    <h1 style="font-size:42px;margin-bottom:16px;">审核完成</h1>
                    <p style="font-size:18px;">可以关闭此页面，或重新开始。</p>
                    <button onclick="location.reload()" style="margin-top:24px;padding:12px 24px;font-size:16px;background:#28a745;color:#fff;border:none;border-radius:6px;cursor:pointer;">
                        重新开始
                    </button>
                </div>
            `;
        }, 300);
    } catch (error) {
        console.error(error);
        alert('提交失败: ' + error.message);
        submitBtn.disabled = false;
        submitExitBtn.disabled = false;
    }
}

async function loadTotalEpisodes() {
    try {
        const response = await fetch('/api/total-episodes');
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.message || '加载总数失败');
        }
        updateQueueModeMeta(data.queue_mode, data.total_items ?? data.total_episodes, data.item_label);
    } catch (error) {
        console.error(error);
        const info = document.getElementById('totalEpisodesInfo');
        if (info) {
            info.textContent = '（无法获取总数）';
            info.style.color = '#f44336';
        }
    }
}

function setStartOffset() {
    const input = document.getElementById('startOffset');
    const offset = parseInt(input.value, 10);
    if (Number.isNaN(offset) || offset < 0) {
        alert('请输入有效的起始位置（非负整数）');
        return;
    }
    currentOffset = offset;
    localStorage.setItem('sanity_check_offset', String(offset));
    cancelPreload().finally(() => loadEpisodes(true));
}

document.addEventListener('DOMContentLoaded', () => {
    loadTotalEpisodes();

    const savedOffset = localStorage.getItem('sanity_check_offset');
    if (savedOffset) {
        currentOffset = parseInt(savedOffset, 10) || 0;
        const input = document.getElementById('startOffset');
        if (input) input.value = String(currentOffset);
    }

    window.addEventListener('beforeunload', () => {
        if (preloadAbortController) {
            preloadAbortController.abort();
        }
    });
});
