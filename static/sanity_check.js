// 全局状态
let episodes = [];
let markedFrames = new Set(); // 存储标记的帧 ID: "episodeId_frameIndex"
let sessionId = null; // 当前审核会话 ID
let reviewedEpisodeIds = new Set(); // 记录所有加载的 episode IDs
let currentOffset = 0; // 当前起始位置

// 预加载相关
let preloadedBatch = null; // 预加载的下一批数据
let isPreloading = false; // 是否正在预加载
let preloadAbortController = null; // 用于取消预加载请求

// 生成或获取用户 ID
function getUserId() {
    let userId = localStorage.getItem('userId');
    if (!userId) {
        userId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('userId', userId);
    }
    return userId;
}

// 生成会话 ID
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// ⭐ 设置起始位置
function setStartOffset() {
    const input = document.getElementById('startOffset');
    const offset = parseInt(input.value);
    
    if (isNaN(offset) || offset < 0) {
        alert('请输入有效的起始位置（非负整数）');
        return;
    }
    
    currentOffset = offset;
    localStorage.setItem('currentOffset', offset);
    console.log(`设置起始位置: ${offset}`);
    
    // 清空页面并加载新批次
    loadEpisodes(true);
}

// ⭐ 简化版：删除所有reviewing相关函数

// 重置进度（从头开始）
function resetProgress() {
    currentOffset = 0;
    localStorage.setItem('sanity_check_offset', '0');
    preloadedBatch = null; // 清空预加载缓存
    const offsetInput = document.getElementById('startOffset');
    if (offsetInput) offsetInput.value = 0;
    console.log('进度已重置到 0');
}

// ⭐ 预加载下一批数据（后台进行，简化版）
async function preloadNextBatch() {
    if (isPreloading) {
        console.log('已经在预加载中，跳过');
        return;
    }
    
    isPreloading = true;
    preloadAbortController = new AbortController();
    
    // 更新UI状态
    const preloadStatus = document.getElementById('preloadStatus');
    if (preloadStatus) {
        preloadStatus.textContent = '🔄 正在预加载下一批...';
    }
    
    try {
        const userId = getUserId();
        const nextOffset = currentOffset; // 使用当前的offset（已经是下一批的offset）
        
        console.log(`🔄 开始预加载下一批 (offset=${nextOffset}, 当前时间=${new Date().toLocaleTimeString()})...`);
        const preloadStartTime = Date.now();
        
        const response = await fetch(
            `/api/episodes/sequential?user_id=${userId}&limit=210&offset=${nextOffset}`,
            { signal: preloadAbortController.signal }
        );
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        const preloadTime = Date.now() - preloadStartTime;
        
        if (!data || !data.success || !data.episodes) {
            throw new Error('预加载数据格式错误');
        }
        
        // 保存预加载的数据
        preloadedBatch = {
            data: data,
            offset: nextOffset,
            timestamp: Date.now()
        };
        
        console.log(`✓✓✓ 预加载完成: ${data.episodes.length} 个episodes (offset=${nextOffset}, 耗时 ${(preloadTime/1000).toFixed(1)}s)`);
        console.log(`📦 预加载数据已缓存，准备好随时使用`);
        
        // 调试：检查有多少 episodes 包含标注信息
        const annotatedCount = data.episodes.filter(ep => ep.annotation && ep.annotation.has_annotation).length;
        if (annotatedCount > 0) {
            console.log(`🏷️ 本批次包含 ${annotatedCount} 个已标注的 episodes`);
            // 显示第一个已标注的 episode 作为示例
            const firstAnnotated = data.episodes.find(ep => ep.annotation && ep.annotation.has_annotation);
            console.log('示例已标注 episode:', firstAnnotated.episode_id, firstAnnotated.annotation);
        } else {
            console.log('⚠️ 本批次没有已标注的 episodes');
        }
        
        // 更新UI状态
        if (preloadStatus) {
            preloadStatus.textContent = `✓ 下一批已准备就绪 (${data.episodes.length}个episodes)`;
        }
        
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('预加载被取消');
        } else {
            console.warn('预加载失败:', error.message);
        }
        preloadedBatch = null;
        
        // 更新UI状态
        if (preloadStatus) {
            preloadStatus.textContent = '';
        }
    } finally {
        isPreloading = false;
        preloadAbortController = null;
    }
}

// 取消预加载
async function cancelPreload() {
    if (isPreloading && preloadAbortController) {
        console.log('取消预加载...');
        preloadAbortController.abort();
    }
    
    // 如果有预加载的数据，清理reviewing标记
    if (preloadedBatch && preloadedBatch.preloadSessionId) {
        try {
            await fetch('/api/sanity-check/cancel-review', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: preloadedBatch.preloadSessionId })
            });
            console.log('已清理预加载的reviewing标记');
        } catch (error) {
            console.warn('清理预加载reviewing标记失败:', error);
        }
    }
    
    preloadedBatch = null;
    
    // 清除UI状态
    const preloadStatus = document.getElementById('preloadStatus');
    if (preloadStatus) {
        preloadStatus.textContent = '';
    }
}

// 加载 episodes（使用顺序加载，网络盘优化）
async function loadEpisodes(clearMarks = true) {
    const submitBtn = document.getElementById('submitBtn');
    const submitExitBtn = document.getElementById('submitExitBtn');
    const loading = document.getElementById('loading');
    const container = document.getElementById('episodesContainer');
    
    if (clearMarks) {
        markedFrames.clear();
        reviewedEpisodeIds.clear();
    }
    
    submitBtn.disabled = true;
    submitExitBtn.disabled = true;
    loading.style.display = 'block';
    container.innerHTML = '';
    
    try {
        const userId = getUserId();
        let data;
        let usingPreloadedData = false;
        
        // 🚀 检查是否有预加载的数据
        console.log(`🔍 检查预加载: preloadedBatch=${preloadedBatch ? `存在(offset=${preloadedBatch.offset})` : '不存在'}, currentOffset=${currentOffset}`);
        
        if (preloadedBatch && preloadedBatch.offset === currentOffset) {
            console.log(`⚡⚡⚡ 使用预加载数据 (offset=${currentOffset})，即时显示！`);
            data = preloadedBatch.data;
            usingPreloadedData = true;
            preloadedBatch = null; // 清空已使用的预加载数据
        } else {
            // 如果没有预加载或offset不匹配，正常加载
            if (preloadedBatch) {
                console.log(`⚠️ 预加载数据不匹配 (预加载offset=${preloadedBatch.offset}, 需要offset=${currentOffset})，重新加载`);
                cancelPreload(); // 取消不匹配的预加载
            } else {
                console.log(`📥 没有预加载数据，正常加载`);
            }
            
            console.log(`⭐ 开始顺序加载 episodes (offset=${currentOffset})...`);
            const loadStartTime = Date.now();
            
            // ⭐ 简化版：直接顺序加载，无reviewing检查
            const response = await fetch(`/api/episodes/sequential?user_id=${userId}&limit=210&offset=${currentOffset}`);
            
            if (!response.ok) {
                // 尝试解析错误消息
                try {
                    const errorData = await response.json();
                    if (errorData.message) {
                        throw new Error(errorData.message);
                    }
                } catch (e) {
                    // 如果无法解析JSON，使用HTTP状态信息
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            data = await response.json();
            const loadTime = Date.now() - loadStartTime;
            console.log(`✓ Sequential API 返回 (耗时 ${(loadTime/1000).toFixed(1)}s):`, data);
        }
        
        if (!data || !data.success) {
            throw new Error(data?.message || 'API 返回 success=false');
        }
        
        if (!data.episodes || !Array.isArray(data.episodes)) {
            throw new Error('API 未返回 episodes 数组');
        }
        
        episodes = data.episodes;

        // 诊断：如果 0 个 episodes 且有失败原因，显示给用户
        if (episodes.length === 0 && data.debug_fail_reasons) {
            const reasons = data.debug_fail_reasons.join('\n');
            console.error('所有 episodes 加载失败，原因:\n' + reasons);
            alert('本批次 0 个 episodes 加载成功。\n\n失败原因（前5条）:\n' + reasons);
        }

        // 保存下一个偏移量
        if (data.next_offset !== undefined) {
            currentOffset = data.next_offset;
            localStorage.setItem('sanity_check_offset', currentOffset.toString());
        }
        
        console.log(`✓ 成功加载 ${episodes.length} 个 episodes (下一批从 ${currentOffset} 开始，共 ${data.total_episodes || '?'} 个)`);
        
        // 显示性能数据
        if (data.performance) {
            const perf = data.performance;
            const displayOffset = currentOffset - episodes.length; // 显示本批次的起始offset
            const displayLimit = episodes.length; // 实际加载的数量
            console.log(`\n${'='.repeat(60)}`);
            console.log(`⏱️  后端性能分析 (offset=${displayOffset}, limit=${displayLimit})`);
            console.log(`${'='.repeat(60)}`);
            console.log(`  加载数据集配置:      ${(perf.load_datasets || 0).toFixed(3)}s`);
            console.log(`  收集元数据:          ${(perf.collect_metadata || 0).toFixed(3)}s`);
            console.log(`  数据库查询标注:      ${(perf.db_query_annotations || 0).toFixed(3)}s`);
            console.log(`  构建标注状态:        ${(perf.build_annotation_status || 0).toFixed(3)}s`);
            console.log(`  Zarr读取图像:        ${(perf.zarr_read_total || 0).toFixed(3)}s  ⚠ 网络I/O瓶颈`);
            console.log(`  图像编码(WebP/B64):  ${(perf.image_encode_total || 0).toFixed(3)}s  ⚠ CPU密集`);
            console.log(`  其他操作:            ${(perf.other || 0).toFixed(3)}s`);
            console.log(`  ${'─'.repeat(58)}`);
            console.log(`  总耗时:              ${(perf.total || 0).toFixed(3)}s`);
            console.log(`${'='.repeat(60)}\n`);
        }
        
        // 更新输入框的值
        const offsetInput = document.getElementById('startOffset');
        if (offsetInput && currentOffset > 0) {
            offsetInput.value = currentOffset;
        }
        
        // 将Sequential API返回的字段映射为前端期望的格式
        episodes = episodes.map(ep => ({
            id: ep.episode_id,
            name: ep.episode_name,
            dataset: ep.dataset_name,
            episode_index: ep.episode_index,
            num_frames: ep.num_frames,
            start_idx: ep.start_idx,
            images: ep.images,
            frame_indices: ep.frame_indices,
            annotation: ep.annotation  // 传递标注信息
        }));
        
        // 后端API已经在选择episodes时就标记为reviewing了，无需前端再次标记
        console.log(`✓ 后端已标记 ${episodes.length} 个episodes为reviewing`);
        
        document.getElementById('loadedCount').textContent = 
            `正在渲染 ${episodes.length} 个 episodes...`;
        
        // ⭐ 分批渲染，避免一次性渲染导致卡顿
        let renderedCount = 0;
        const renderBatchSize = 30;  // 210个episodes，分7批渲染
        
        for (let i = 0; i < episodes.length; i += renderBatchSize) {
            const batch = episodes.slice(i, Math.min(i + renderBatchSize, episodes.length));
            
            // 渲染这一批
            batch.forEach(episode => {
                renderEpisode(episode, episode);  // episode对象已包含所有数据
                renderedCount++;
            });
            
            // 更新进度
            document.getElementById('loadedCount').textContent = 
                `正在渲染 ${renderedCount}/${episodes.length} 个 episodes...`;
            
            // 让浏览器有时间渲染，避免卡顿（增加延迟让用户看到渐进式加载）
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        document.getElementById('loadedCount').textContent = 
            `已加载 ${episodes.length} 个 episodes`;
        
        submitBtn.disabled = false;
        submitExitBtn.disabled = false;
        
        // 更新统计
        updateStats();
        
        // 🚀 渲染完成后，立即开始预加载下一批（后台进行）
        if (data.has_more && currentOffset > 0) {
            console.log(`💡 立即开始预加载下一批 (offset=${currentOffset})...`);
            setTimeout(() => preloadNextBatch(), 100); // 极短延迟，几乎立即开始
        } else {
            console.log('✓ 已加载所有数据，无需预加载');
        }
        
    } catch (error) {
        console.error('加载失败 - 详细错误:', error);
        console.error('错误堆栈:', error.stack);
        const errorMsg = error.message || error.toString() || '未知错误';
        alert('加载失败: ' + errorMsg + '\n请打开浏览器控制台查看详细信息');
    } finally {
        loading.style.display = 'none';
    }
}

// ⭐ 流式加载 episode 数据（加载一个显示一个）
async function loadAllEpisodeDataParallel() {
    const container = document.getElementById('episodesContainer');
    const userId = getUserId();
    
    console.log(`开始流式加载 ${episodes.length} 个 episodes...`);
    
    let successCount = 0;
    let failCount = 0;
    let loadedCount = 0;
    
    // 使用批量API，每批 50 个（最大化减少API调用次数）
    const batchSize = 20;
    
    for (let i = 0; i < episodes.length; i += batchSize) {
        const batch = episodes.slice(i, Math.min(i + batchSize, episodes.length));
        const batchEpisodeIds = batch.map(ep => ep.id);
        
        try {
            // 一次性批量加载多个episode
            const response = await fetch('/api/episodes/batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    episode_ids: batchEpisodeIds,
                    user_id: userId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const batchData = await response.json();
            
            if (!batchData.success) {
                throw new Error(batchData.message || 'Batch API 失败');
            }
            
            // 创建一个map方便查找
            const dataMap = new Map();
            batchData.episodes.forEach(epData => {
                dataMap.set(epData.episode_id, epData);
            });
            
            // 立即渲染这一批成功加载的 episodes
            batch.forEach((episode, batchIdx) => {
                const globalIdx = i + batchIdx;
                loadedCount++;
                
                const data = dataMap.get(episode.id);
                if (data && data.success) {
                    renderEpisode(episode, data);
                    successCount++;
                } else {
                    failCount++;
                    console.warn(`Episode #${globalIdx} (${episode.id}) 未返回数据`);
                }
            });
            
        } catch (err) {
            console.error(`批量加载第 ${i}-${i+batchSize} 批失败:`, err);
            failCount += batch.length;
            loadedCount += batch.length;
        }
        
        // 更新进度
        document.getElementById('loadedCount').textContent = 
            `已加载 ${loadedCount} / ${episodes.length} 个 episodes (${successCount} 成功, ${failCount} 失败)`;
        
        // 更新统计
        updateStats();
    }
    
    console.log(`流式加载完成: ${successCount} 成功, ${failCount} 失败`);
    
    if (successCount === 0) {
        throw new Error(`所有 episodes 都加载失败`);
    }
}

// 渲染单个 episode
function renderEpisode(episode, data) {
    try {
        const container = document.getElementById('episodesContainer');
        
        // 验证数据完整性
        if (!data.images || !Array.isArray(data.images) || data.images.length === 0) {
            console.warn(`Episode ${episode.id} 没有图像数据`, data);
            return;
        }
        
        if (!data.frame_indices || !Array.isArray(data.frame_indices)) {
            console.warn(`Episode ${episode.id} 缺少 frame_indices`, data);
            return;
        }
        
        const episodeBlock = document.createElement('div');
        episodeBlock.className = 'episode-block';
        episodeBlock.dataset.episodeId = episode.id;
        
        // 检查是否有标注状态
        if (data.annotation && data.annotation.has_annotation) {
            console.log(`Episode ${episode.id} 有标注状态:`, data.annotation);
            const markType = data.annotation.mark_type;
            episodeBlock.classList.add('annotated', markType);
            
            // 添加标注状态标签
            const statusLabel = document.createElement('div');
            statusLabel.className = `annotation-status ${markType}`;
            
            if (markType === 'alright') {
                statusLabel.innerHTML = '✓ 已标注：Alright';
            } else if (markType === 'bad') {
                const badCount = data.annotation.bad_frames ? data.annotation.bad_frames.length : 0;
                statusLabel.innerHTML = `✗ 已标注：Bad Frame (${badCount}个问题帧)`;
            }
            
            episodeBlock.appendChild(statusLabel);
        } else {
            // 调试：查看是否有 annotation 字段
            if (data.annotation) {
                console.log(`Episode ${episode.id} annotation 存在但 has_annotation=false:`, data.annotation);
            }
        }
        
        // Header
        const header = document.createElement('div');
        header.className = 'episode-header';
        
        const title = document.createElement('div');
        title.className = 'episode-title';
        title.innerHTML = `
            <span class="dataset-badge">${episode.dataset || 'Unknown'}</span>
            <span>${episode.name || episode.id}</span>
        `;
        
        const info = document.createElement('div');
        info.className = 'episode-info';
        info.textContent = `Episode #${episode.episode_index || '?'} | ${data.num_frames || '?'} frames`;
        
        header.appendChild(title);
        header.appendChild(info);
        
        // Frames container
        const framesContainer = document.createElement('div');
        framesContainer.className = 'frames-container';
        
        // 渲染每一帧
        data.images.forEach((img, idx) => {
            const frameWrapper = document.createElement('div');
            frameWrapper.className = 'frame-wrapper';
            
            const absoluteFrameIndex = data.frame_indices[idx] || idx;
            const startIdx = data.start_idx || 0;
            const relativeFrameIndex = absoluteFrameIndex - startIdx;  // 相对于episode开头的帧号
            
            const frameId = `${episode.id}_${relativeFrameIndex}`;
            frameWrapper.dataset.frameId = frameId;
            frameWrapper.dataset.absoluteIndex = absoluteFrameIndex;  // 保存绝对索引用于其他用途
            
            // 检查是否已标注（来自之前的标注）
            if (data.annotation && data.annotation.mark_type === 'bad' && data.annotation.bad_frames) {
                // 检查该帧是否在 bad_frames 列表中
                if (data.annotation.bad_frames.includes(relativeFrameIndex)) {
                    markedFrames.add(frameId);
                    frameWrapper.classList.add('marked');
                }
            }
            
            // 检查是否在当前会话中标注
            if (markedFrames.has(frameId)) {
                frameWrapper.classList.add('marked');
            }
            
            // 点击切换标注状态
            frameWrapper.onclick = () => toggleMark(frameId, frameWrapper);
            
            const frameIndexLabel = document.createElement('div');
            frameIndexLabel.className = 'frame-index';
            frameIndexLabel.textContent = `Frame ${relativeFrameIndex}`;
            
            const imgElement = document.createElement('img');
            imgElement.src = img;  // 已经包含 data URL 前缀
            imgElement.alt = `Frame ${idx}`;
            
            frameWrapper.appendChild(frameIndexLabel);
            frameWrapper.appendChild(imgElement);
            framesContainer.appendChild(frameWrapper);
        });
        
        episodeBlock.appendChild(header);
        episodeBlock.appendChild(framesContainer);
        container.appendChild(episodeBlock);
        
    } catch (error) {
        console.error(`渲染 episode ${episode.id} 失败:`, error, data);
    }
}

// 切换标注状态
function toggleMark(frameId, element) {
    if (markedFrames.has(frameId)) {
        markedFrames.delete(frameId);
        element.classList.remove('marked');
    } else {
        markedFrames.add(frameId);
        element.classList.add('marked');
    }
    
    updateStats();
}

// 更新统计信息
function updateStats() {
    const totalEpisodes = episodes.length;
    const totalImages = document.querySelectorAll('.frame-wrapper').length;
    const markedImages = markedFrames.size;
    
    document.getElementById('totalEpisodes').textContent = totalEpisodes;
    document.getElementById('totalImages').textContent = totalImages;
    document.getElementById('markedImages').textContent = markedImages;
}

// 提交审核数据（通用函数）
async function submitReviewData() {
    // 准备标注数据（有问题的帧）
    const markedFramesData = Array.from(markedFrames).map(frameId => {
        const lastUnderscore = frameId.lastIndexOf('_');
        const episodeId = frameId.substring(0, lastUnderscore);
        const frameIndex = frameId.substring(lastUnderscore + 1);
        const episode = episodes.find(e => e.id === episodeId);

        return {
            episode_id: episodeId,
            dataset: episode ? episode.dataset : '',
            episode_name: episode ? episode.name : '',
            episode_index: episode ? episode.episode_index : 0,
            frame_index: parseInt(frameIndex)
        };
    });
    
        // 准备已审核的 episodes 列表（所有加载的 episodes）
        const reviewedEpisodesData = episodes.map(episode => {
            // 检查该 episode 是否有标注
            const hasAnnotation = Array.from(markedFrames).some(frameId => 
                frameId.startsWith(episode.id + '_')
            );
            
            return {
                episode_id: episode.id,
                dataset: episode.dataset,
                episode_name: episode.name,
                episode_index: episode.episode_index,
                has_annotation: hasAnnotation
            };
        });
    
    // 提交到服务器
    const response = await fetch('/api/sanity-check/submit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            marked_frames: markedFramesData,
            reviewed_episodes: reviewedEpisodesData,
            user_id: getUserId(),
            session_id: sessionId
        })
    });
    
    const data = await response.json();
    
    if (!data.success) {
        throw new Error(data.message || '提交失败');
    }
    
    console.log(`已提交 ${data.total_count} 个标注 (${data.alright_count} 正常, ${data.bad_count} 有问题)`);
    
    return data;
}

// 提交标注并加载下一批
async function submitAndLoadNext() {
    const submitBtn = document.getElementById('submitBtn');
    const submitExitBtn = document.getElementById('submitExitBtn');
    
    submitBtn.disabled = true;
    submitExitBtn.disabled = true;
    
    try {
        await submitReviewData();
        
        // ⭐ 不要取消预加载！loadEpisodes会自动使用预加载的数据（如果可用）
        await loadEpisodes(true);  // clearMarks = true
        
    } catch (error) {
        console.error('提交或加载失败:', error);
        alert('操作失败: ' + error.message);
        submitBtn.disabled = false;
        submitExitBtn.disabled = false;
    }
}

// 提交标注并退出
async function submitAndExit() {
    const submitBtn = document.getElementById('submitBtn');
    const submitExitBtn = document.getElementById('submitExitBtn');
    
    submitBtn.disabled = true;
    submitExitBtn.disabled = true;
    
    // 取消任何正在进行的预加载
    await cancelPreload();
    
    try {
        await submitReviewData();
        
        // 显示感谢消息
        alert('提交成功！感谢您的审核工作。');
        
        // 清空会话
        sessionId = null;
        
        // 关闭窗口或显示完成页面
        window.close();
        
        // 如果无法关闭（浏览器限制），则显示完成信息
        setTimeout(() => {
            document.body.innerHTML = `
                <div style="display: flex; align-items: center; justify-content: center; height: 100vh; flex-direction: column; color: #e0e0e0;">
                    <h1 style="font-size: 48px; margin-bottom: 20px;">✓ 审核完成</h1>
                    <p style="font-size: 20px;">感谢您的审核工作！您可以关闭此页面。</p>
                    <button onclick="location.reload()" style="margin-top: 30px; padding: 15px 30px; font-size: 16px; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer;">
                        重新开始审核
                    </button>
                </div>
            `;
        }, 500);
        
    } catch (error) {
        console.error('提交失败:', error);
        alert('提交失败: ' + error.message);
        submitBtn.disabled = false;
        submitExitBtn.disabled = false;
    }
}

// 加载总 episodes 数
async function loadTotalEpisodes() {
    try {
        const response = await fetch('/api/total-episodes');
        const data = await response.json();
        
        if (data.success) {
            const infoElement = document.getElementById('totalEpisodesInfo');
            infoElement.textContent = `（总共: ${data.total_episodes} 个episodes）`;
            infoElement.style.color = '#4CAF50';
            
            // 更新输入框的 max 属性
            const input = document.getElementById('startOffset');
            input.max = data.total_episodes - 1;
            
            console.log(`✓ 总共 ${data.total_episodes} 个episodes，跨 ${data.datasets.length} 个数据集`);
            data.datasets.forEach(ds => {
                console.log(`  - ${ds.name}: ${ds.count} 个episodes`);
            });
        }
    } catch (error) {
        console.error('加载总episodes数失败:', error);
        document.getElementById('totalEpisodesInfo').textContent = '（无法获取总数）';
        document.getElementById('totalEpisodesInfo').style.color = '#f44336';
    }
}

// 页面加载完成后的初始化
document.addEventListener('DOMContentLoaded', () => {
    console.log('Sanity Check page ready');
    
    // 加载总 episodes 数
    loadTotalEpisodes();
    
    // 读取保存的offset
    const savedOffset = localStorage.getItem('sanity_check_offset');
    if (savedOffset) {
        currentOffset = parseInt(savedOffset);
        document.getElementById('startOffset').value = currentOffset;
        console.log(`恢复上次进度: offset=${currentOffset}`);
    }
    
    // ⭐ 不自动加载，等待用户点击"设置并开始"按钮
    console.log('请输入起始位置并点击"设置并开始"按钮');
    
    // 页面关闭时的清理（保留以防万一）
    window.addEventListener('beforeunload', (e) => {
        // 简化版：无需取消reviewing
        console.log('页面关闭');
    });
    
    // 页面隐藏时的处理（简化版，无需特殊处理）
    document.addEventListener('visibilitychange', () => {
        console.log('页面可见性变化:', document.hidden ? '隐藏' : '显示');
    });
});

