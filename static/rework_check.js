/**
 * 返工页：仅加载仍为 BAD_FRAMES/BAD_FRAME 的 episode。
 * 帧状态循环：ok → bad_box → not_clear → ok（默认 ok，无需点击）
 */

const FRAME_STATE_SEP = '\u0001';
const LABELS = {
    ok: '',
    bad_box: '手部可见，标注框错误',
    not_clear: '手部非清晰可见',
};

let episodes = [];
/** @type {Map<string, 'bad_box'|'not_clear'>} 仅存储非 ok */
const frameStates = new Map();
let currentOffset = 0;

let preloadedBatch = null;
let isPreloading = false;
let preloadAbortController = null;

function frameKey(episodeId, relativeFrameIndex) {
    return episodeId + FRAME_STATE_SEP + String(relativeFrameIndex);
}

function getState(episodeId, relativeFrameIndex) {
    return frameStates.get(frameKey(episodeId, relativeFrameIndex)) || 'ok';
}

function setFrameState(episodeId, relativeFrameIndex, state) {
    const k = frameKey(episodeId, relativeFrameIndex);
    if (state === 'ok') {
        frameStates.delete(k);
    } else {
        frameStates.set(k, state);
    }
}

function nextState(state) {
    if (state === 'ok') return 'bad_box';
    if (state === 'bad_box') return 'not_clear';
    return 'ok';
}

function applyStateToElement(el, state) {
    el.classList.remove('state-ok', 'state-bad_box', 'state-not_clear');
    el.classList.add('state-' + state);
    el.dataset.frameState = state;
    const label = el.querySelector('.frame-mask-label');
    if (label) {
        label.textContent = LABELS[state] || '';
    }
}

function getUserId() {
    let userId = localStorage.getItem('userId');
    if (!userId) {
        userId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('userId', userId);
    }
    return userId;
}

function setStartOffset() {
    const input = document.getElementById('startOffset');
    const offset = parseInt(input.value, 10);
    if (isNaN(offset) || offset < 0) {
        alert('请输入有效的起始序号（非负整数）');
        return;
    }
    currentOffset = offset;
    localStorage.setItem('rework_check_offset', String(offset));
    loadEpisodes(true);
}

async function preloadNextBatch() {
    if (isPreloading) return;
    isPreloading = true;
    preloadAbortController = new AbortController();
    const preloadStatus = document.getElementById('preloadStatus');
    if (preloadStatus) preloadStatus.textContent = '🔄 正在预加载下一批…';

    try {
        const userId = getUserId();
        const nextOffset = currentOffset;
        const response = await fetch(
            `/api/rework/episodes?user_id=${encodeURIComponent(userId)}&limit=210&offset=${nextOffset}`,
            { signal: preloadAbortController.signal }
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (!data || !data.success || !data.episodes) throw new Error('预加载数据格式错误');
        preloadedBatch = { data, offset: nextOffset, timestamp: Date.now() };
        if (preloadStatus) {
            preloadStatus.textContent = `✓ 下一批已就绪（${data.episodes.length} 条）`;
        }
    } catch (error) {
        if (error.name !== 'AbortError') {
            console.warn('预加载失败:', error.message);
        }
        preloadedBatch = null;
        const preloadStatus = document.getElementById('preloadStatus');
        if (preloadStatus) preloadStatus.textContent = '';
    } finally {
        isPreloading = false;
        preloadAbortController = null;
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

function initialStateFromRework(episodeId, relativeFrameIndex, reworkAnn) {
    if (!reworkAnn) return 'ok';
    const bb = reworkAnn.bad_box || [];
    const nc = reworkAnn.not_clear || [];
    if (bb.includes(relativeFrameIndex)) return 'bad_box';
    if (nc.includes(relativeFrameIndex)) return 'not_clear';
    return 'ok';
}

async function loadEpisodes(clearStates) {
    const submitBtn = document.getElementById('submitBtn');
    const submitExitBtn = document.getElementById('submitExitBtn');
    const loading = document.getElementById('loading');
    const container = document.getElementById('episodesContainer');

    if (clearStates) {
        frameStates.clear();
    }

    submitBtn.disabled = true;
    submitExitBtn.disabled = true;
    loading.style.display = 'block';
    container.innerHTML = '';

    try {
        const userId = getUserId();
        let data;
        if (preloadedBatch && preloadedBatch.offset === currentOffset) {
            data = preloadedBatch.data;
            preloadedBatch = null;
        } else {
            await cancelPreload();
            const response = await fetch(
                `/api/rework/episodes?user_id=${encodeURIComponent(userId)}&limit=210&offset=${currentOffset}`
            );
            if (!response.ok) {
                let msg = `HTTP ${response.status}`;
                try {
                    const err = await response.json();
                    if (err.message) msg = err.message;
                } catch (e) { /* ignore */ }
                throw new Error(msg);
            }
            data = await response.json();
        }

        if (!data.success) {
            throw new Error(data.message || 'API success=false');
        }
        if (!Array.isArray(data.episodes)) {
            throw new Error('API 未返回 episodes 数组');
        }

        if (data.next_offset !== undefined) {
            currentOffset = data.next_offset;
            localStorage.setItem('rework_check_offset', String(currentOffset));
        }

        episodes = data.episodes.map((ep) => ({
            id: ep.episode_id,
            name: ep.episode_name,
            dataset: ep.dataset_name,
            episode_index: ep.episode_index,
            num_frames: ep.num_frames,
            start_idx: ep.start_idx || 0,
            images: ep.images,
            frame_indices: ep.frame_indices,
            legacy_bad_frames: ep.legacy_bad_frames || [],
            rework_annotation: ep.rework_annotation || null,
            stage1_pending: !!ep.stage1_pending,
        }));

        if (episodes.length === 0 && data.debug_fail_reasons) {
            alert('本批 0 条加载成功。\n\n原因示例:\n' + data.debug_fail_reasons.join('\n'));
        }

        document.getElementById('loadedCount').textContent =
            data.total_episodes === 0
                ? '当前没有待返工条目（无 BAD_FRAMES 记录）'
                : `已加载 ${episodes.length} 条 · 队列共 ${data.total_episodes} 条`;

        const offsetInput = document.getElementById('startOffset');
        if (offsetInput && currentOffset > 0) {
            offsetInput.value = currentOffset;
        }

        const renderBatchSize = 30;
        let rendered = 0;
        for (let i = 0; i < episodes.length; i += renderBatchSize) {
            const slice = episodes.slice(i, Math.min(i + renderBatchSize, episodes.length));
            slice.forEach((episode) => {
                const raw = data.episodes.find((e) => e.episode_id === episode.id);
                renderEpisode(episode, raw || episode);
                rendered++;
            });
            document.getElementById('loadedCount').textContent = `正在渲染 ${rendered}/${episodes.length}…`;
            await new Promise((r) => setTimeout(r, 80));
        }

        document.getElementById('loadedCount').textContent =
            data.total_episodes === 0
                ? '没有待返工数据'
                : `已展示 ${episodes.length} 条 episode`;

        submitBtn.disabled = true;
        submitExitBtn.disabled = true;

        const totalBad = data.total_episodes || 0;
        if (totalBad > 0) {
            submitBtn.disabled = false;
            submitExitBtn.disabled = false;
        }

        updateStats();

        if (data.has_more && currentOffset > 0) {
            setTimeout(() => preloadNextBatch(), 120);
        }
    } catch (error) {
        console.error(error);
        alert('加载失败: ' + (error.message || error));
    } finally {
        loading.style.display = 'none';
    }
}

function renderEpisode(episode, data) {
    const container = document.getElementById('episodesContainer');

    if (data.stage1_pending) {
        const block = document.createElement('div');
        block.className = 'episode-block stage1-pending legacy-bad';
        block.dataset.episodeId = episode.id;

        const header = document.createElement('div');
        header.className = 'episode-header';
        const title = document.createElement('div');
        title.className = 'episode-title';
        title.innerHTML = `<span class="dataset-badge">${episode.dataset || ''}</span><span>${episode.name || episode.id}</span>`;
        const info = document.createElement('div');
        info.className = 'episode-info';
        info.textContent = `${data.num_frames != null ? data.num_frames : '?'} frames`;
        header.appendChild(title);
        header.appendChild(info);

        const pending = document.createElement('div');
        pending.className = 'stage1-pending-label';
        pending.textContent = 'Stage1 未完成，无法出图；可跳过或稍后重试。';

        block.appendChild(header);
        block.appendChild(pending);
        container.appendChild(block);
        return;
    }

    if (!data.images || !data.images.length || !data.frame_indices) {
        console.warn('跳过无图 episode', episode.id);
        return;
    }

    const block = document.createElement('div');
    block.className = 'episode-block legacy-bad';
    block.dataset.episodeId = episode.id;

    const header = document.createElement('div');
    header.className = 'episode-header';
    const title = document.createElement('div');
    title.className = 'episode-title';
    title.innerHTML = `<span class="dataset-badge">${episode.dataset || ''}</span><span>${episode.name || episode.id}</span>`;
    const info = document.createElement('div');
    info.className = 'episode-info';
    info.textContent = `${data.num_frames != null ? data.num_frames : '?'} frames`;

    const legacyTag = document.createElement('div');
    legacyTag.className = 'legacy-tag';
    const lb = data.legacy_bad_frames || [];
    legacyTag.textContent =
        lb.length > 0 ? `曾标问题帧: ${lb.join(', ')}` : '历史标记为本条有问题';

    header.appendChild(title);
    header.appendChild(info);
    block.appendChild(header);
    block.appendChild(legacyTag);

    const framesContainer = document.createElement('div');
    framesContainer.className = 'frames-container';

    const reworkAnn = data.rework_annotation || null;

    data.images.forEach((imgSrc, idx) => {
        const absIdx = data.frame_indices[idx];
        const startIdx = data.start_idx || 0;
        const relativeFrameIndex = absIdx - startIdx;

        const wrap = document.createElement('div');
        wrap.className = 'frame-wrapper';
        wrap.dataset.episodeId = episode.id;
        wrap.dataset.relativeFrameIndex = String(relativeFrameIndex);

        const idxLabel = document.createElement('div');
        idxLabel.className = 'frame-index';
        idxLabel.textContent = '帧 ' + relativeFrameIndex;

        const img = document.createElement('img');
        img.src = imgSrc;
        img.alt = 'frame ' + relativeFrameIndex;

        const mask = document.createElement('div');
        mask.className = 'frame-mask';
        const maskLabel = document.createElement('div');
        maskLabel.className = 'frame-mask-label';
        mask.appendChild(maskLabel);

        wrap.appendChild(idxLabel);
        wrap.appendChild(img);
        wrap.appendChild(mask);

        const st = initialStateFromRework(episode.id, relativeFrameIndex, reworkAnn);
        setFrameState(episode.id, relativeFrameIndex, st);
        applyStateToElement(wrap, st);

        wrap.addEventListener('click', () => {
            const cur = getState(episode.id, relativeFrameIndex);
            const nxt = nextState(cur);
            setFrameState(episode.id, relativeFrameIndex, nxt);
            applyStateToElement(wrap, nxt);
            updateStats();
        });

        framesContainer.appendChild(wrap);
    });

    block.appendChild(framesContainer);
    container.appendChild(block);
}

function updateStats() {
    let ok = 0;
    let bad = 0;
    let nc = 0;
    for (const ep of episodes) {
        if (ep.stage1_pending || !ep.frame_indices) continue;
        for (let i = 0; i < ep.frame_indices.length; i++) {
            const absIdx = ep.frame_indices[i];
            const rel = absIdx - (ep.start_idx || 0);
            const s = getState(ep.id, rel);
            if (s === 'ok') ok++;
            else if (s === 'bad_box') bad++;
            else nc++;
        }
    }
    document.getElementById('statOk').textContent = ok;
    document.getElementById('statBadBox').textContent = bad;
    document.getElementById('statNotClear').textContent = nc;
    document.getElementById('totalEpisodes').textContent = episodes.length;
}

async function submitReworkPayload() {
    const toSave = episodes.filter(
        (ep) => !ep.stage1_pending && ep.frame_indices && ep.frame_indices.length > 0
    );
    const payloadEpisodes = toSave.map((ep) => {
        const bad_box = [];
        const not_clear = [];
        for (const absIdx of ep.frame_indices) {
            const rel = absIdx - (ep.start_idx || 0);
            const s = getState(ep.id, rel);
            if (s === 'bad_box') bad_box.push(rel);
            else if (s === 'not_clear') not_clear.push(rel);
        }
        return {
            episode_id: ep.id,
            episode_name: ep.name,
            dataset_name: ep.dataset,
            episode_index: ep.episode_index,
            bad_box,
            not_clear,
        };
    });

    const response = await fetch('/api/rework/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ episodes: payloadEpisodes }),
    });
    const data = await response.json();
    if (!data.success) {
        throw new Error(data.message || '提交失败');
    }
    return data;
}

async function submitAndLoadNext() {
    const submitBtn = document.getElementById('submitBtn');
    const submitExitBtn = document.getElementById('submitExitBtn');
    submitBtn.disabled = true;
    submitExitBtn.disabled = true;
    try {
        if (episodes.length > 0) {
            await submitReworkPayload();
        }
        await loadEpisodes(true);
    } catch (e) {
        console.error(e);
        alert('操作失败: ' + e.message);
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
        if (episodes.length > 0) {
            await submitReworkPayload();
        }
        alert('提交成功，感谢。');
        window.close();
        setTimeout(() => {
            document.body.innerHTML =
                '<div style="display:flex;align-items:center;justify-content:center;min-height:100vh;flex-direction:column;background:#141820;color:#e8eaed;"><p style="font-size:18px;">已完成，可关闭页面。</p><button type="button" id="reloadBtn" style="margin-top:20px;padding:12px 24px;border-radius:8px;border:none;background:#2ea043;color:#fff;cursor:pointer;font-size:15px;">重新打开</button></div>';
            const b = document.getElementById('reloadBtn');
            if (b) b.onclick = () => location.reload();
        }, 400);
    } catch (e) {
        alert('提交失败: ' + e.message);
        submitBtn.disabled = false;
        submitExitBtn.disabled = false;
    }
}

async function loadTotal() {
    try {
        const response = await fetch('/api/rework/total');
        const data = await response.json();
        const info = document.getElementById('totalEpisodesInfo');
        if (data.success) {
            info.textContent = `（待返工: ${data.total_episodes} 条）`;
            info.style.color = '#56d364';
            const input = document.getElementById('startOffset');
            if (input && data.total_episodes > 0) {
                input.max = data.total_episodes - 1;
            }
        }
    } catch (e) {
        document.getElementById('totalEpisodesInfo').textContent = '（无法获取待返工数量）';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    loadTotal();
    const saved = localStorage.getItem('rework_check_offset');
    if (saved !== null && saved !== '') {
        currentOffset = parseInt(saved, 10) || 0;
        const input = document.getElementById('startOffset');
        if (input) input.value = currentOffset;
    }
    document.getElementById('setOffsetBtn').addEventListener('click', setStartOffset);
    document.getElementById('submitBtn').addEventListener('click', submitAndLoadNext);
    document.getElementById('submitExitBtn').addEventListener('click', submitAndExit);
});
