// Zarr 数据集查看器 - 前端脚本

let currentEpisodeId = null;
let currentLimit = 50;  // 所有数据集总共随机加载的数量
let userId = null;
let allEpisodes = []; // 所有已加载的episodes

// 预加载相关
let preloadedVideos = new Map(); // 存储预加载的视频: episode_id -> {original: url, rendered: url}
let preloadQueue = []; // 预加载队列
let isPreloading = false;
let maxConcurrentPreload = 2; // 最大并发预加载数（降低以避免带宽占用过多）
let activePreloadCount = 0; // 当前正在预加载的数量

// 翻译预加载相关
let translationQueue = []; // 翻译队列
let activeTranslationCount = 0; // 当前正在翻译的数量
let maxConcurrentTranslation = 2; // 最大并发翻译数
let preloadedTranslations = new Set(); // 已预加载翻译的 episode_id

// 未保存更改跟踪
let savedAnnotationState = {
    issues: [],
    additionalNotes: '',
    submitted: false  // 当前 episode 是否已提交评估
};

let initialAnnotationState = null;  // 页面加载时的初始状态

// 当前翻译数据（用于提交时收集反馈）
let currentTranslations = []; // [{index, original, translation}]

// 审核人姓名
let reviewerName = localStorage.getItem('reviewer_name') || '';

// 生成或获取用户ID
function getUserId() {
    if (!userId) {
        userId = localStorage.getItem('zarr_viewer_user_id');
        if (!userId) {
            // 生成新的UUID
            userId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('zarr_viewer_user_id', userId);
        }
    }
    return userId;
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    // 检查是否需要显示审核人姓名 modal
    if (!reviewerName) {
        showReviewerModal();
    } else {
        updateReviewerDisplay();
    }
    getUserId();  // 确保有用户ID
    loadEpisodes();
    setupEventListeners();
});

// 审核人姓名 Modal 逻辑
function showReviewerModal() {
    const modal = document.getElementById('reviewerModal');
    const input = document.getElementById('reviewerNameInput');
    const btn = document.getElementById('reviewerModalConfirm');
    if (!modal) return;

    modal.style.display = 'flex';
    input.focus();

    input.addEventListener('input', () => {
        btn.disabled = !input.value.trim();
    });

    btn.addEventListener('click', () => {
        confirmReviewerName(input.value.trim());
    });

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && input.value.trim()) {
            confirmReviewerName(input.value.trim());
        }
    });
}

function confirmReviewerName(name) {
    reviewerName = name;
    localStorage.setItem('reviewer_name', name);
    document.getElementById('reviewerModal').style.display = 'none';
    updateReviewerDisplay();
}

function updateReviewerDisplay() {
    const el = document.getElementById('reviewerDisplay');
    if (el && reviewerName) {
        el.textContent = '审核人: ' + reviewerName;
    }
}

// 设置事件监听器
function setupEventListeners() {
    const saveButton = document.getElementById('saveAnnotation');
    if (saveButton) {
        saveButton.addEventListener('click', saveAnnotation);
    }

    // 上一个/下一个按钮
    const prevBtn = document.getElementById('prevEpisodeBtn');
    const nextBtn = document.getElementById('nextEpisodeBtn');
    if (prevBtn) {
        prevBtn.addEventListener('click', navigateToPrevEpisode);
    }
    if (nextBtn) {
        nextBtn.addEventListener('click', navigateToNextEpisode);
    }

    // 键盘快捷键：空格播放暂停
    document.addEventListener('keydown', (e) => {
        // 如果正在输入文本，不触发快捷键
        if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') {
            return;
        }

        if (e.key === ' ') {
            e.preventDefault();
            toggleVideoPlayback();
            return;
        }
    });
}

// 空格键切换播放/暂停
function toggleVideoPlayback() {
    const framesGrid = document.getElementById('framesGrid');
    if (!framesGrid) return;
    const videos = framesGrid.querySelectorAll('.scrub-video');
    if (videos.length === 0) return;

    // 以 original 视频的状态为准
    const primary = videos[0];
    if (primary.paused) {
        // 开始播放前，先同步 rendered 到 original 的时间点
        if (videos.length > 1 && videos[1].duration && isFinite(videos[1].duration)) {
            videos[1].currentTime = Math.min(primary.currentTime, videos[1].duration);
        }
        videos.forEach(v => v.play());
    } else {
        videos.forEach(v => v.pause());
    }
}

// 更新预加载状态显示
function updatePreloadStatus() {
    const statusEl = document.getElementById('preloadStatus');
    if (!statusEl) return;
    
    const total = allEpisodes.length;
    const videoCompleted = preloadedVideos.size;
    const videoRemaining = preloadQueue.length;
    const videoActive = activePreloadCount;
    
    const translationCompleted = preloadedTranslations.size;
    const translationRemaining = translationQueue.length;
    const translationActive = activeTranslationCount;
    
    const parts = [];
    
    // 视频预加载状态
    if (videoActive > 0 || videoRemaining > 0) {
        parts.push(`🎬 视频: ${videoCompleted}/${total}`);
    } else if (videoCompleted > 0) {
        parts.push(`✓ 视频: ${videoCompleted}/${total}`);
    }
    
    // 翻译预加载状态
    if (translationActive > 0 || translationRemaining > 0) {
        parts.push(`📝 翻译: ${translationCompleted}/${total}`);
    } else if (translationCompleted > 0) {
        parts.push(`✓ 翻译: ${translationCompleted}/${total}`);
    }
    
    if (parts.length > 0) {
        statusEl.textContent = parts.join(' | ');
        
        // 如果都完成了，变绿色
        if (videoRemaining === 0 && videoActive === 0 && translationRemaining === 0 && translationActive === 0) {
            statusEl.style.color = '#4CAF50';
        } else {
            statusEl.style.color = '#2196F3';
        }
    } else {
        statusEl.textContent = '';
    }
}

// 预加载视频（单个episode）- 真正下载视频到浏览器缓存
async function preloadVideo(episodeId) {
    if (preloadedVideos.has(episodeId)) {
        return;
    }
    
    activePreloadCount++;
    updatePreloadStatus();
    
    const startTime = Date.now();
    
    try {
        const originalUrl = `/api/episode/${episodeId}/video/original`;
        const renderedUrl = `/api/episode/${episodeId}/video/rendered`;
        
        const total = allEpisodes.length;
        const completed = preloadedVideos.size;
        console.log(`🔄 [${completed + 1}/${total}] 预加载: ${episodeId.substring(0, 30)}...`);
        
        // 创建隐藏的 video 元素来真正下载和缓存视频
        const originalVideo = document.createElement('video');
        const renderedVideo = document.createElement('video');
        
        // 设置预加载属性
        originalVideo.preload = 'auto';
        renderedVideo.preload = 'auto';
        originalVideo.muted = true;
        renderedVideo.muted = true;
        
        // 设置视频源
        originalVideo.src = originalUrl;
        renderedVideo.src = renderedUrl;
        
        // 等待视频加载（rendered 可能需要后端渲染，允许失败）
        const originalPromise = new Promise((resolve, reject) => {
            originalVideo.addEventListener('loadeddata', resolve, { once: true });
            originalVideo.addEventListener('error', reject, { once: true });
            originalVideo.load();
        });
        const renderedPromise = new Promise((resolve, reject) => {
            renderedVideo.addEventListener('loadeddata', resolve, { once: true });
            renderedVideo.addEventListener('error', () => resolve('rendered_failed'), { once: true });
            renderedVideo.load();
        });

        await Promise.all([originalPromise, renderedPromise]);
        
        // 标记为已预加载
        preloadedVideos.set(episodeId, {
            original: originalUrl,
            rendered: renderedUrl,
            originalVideo: originalVideo,
            renderedVideo: renderedVideo,
            timestamp: Date.now()
        });
        
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        const remaining = preloadQueue.length;
        console.log(`✓ [${completed + 1}/${total}] 完成 (${elapsed}s, 剩余 ${remaining})`);
        
    } catch (error) {
        console.warn(`预加载失败 (${episodeId.substring(0, 30)}):`, error.message);
    } finally {
        activePreloadCount--;
        updatePreloadStatus();
        // 继续处理队列
        processPreloadQueue();
    }
}

// 处理预加载队列（并发控制）
function processPreloadQueue() {
    // 如果队列为空，不处理
    if (preloadQueue.length === 0) {
        if (activePreloadCount === 0 && preloadedVideos.size > 0) {
            console.log(`🎉 所有视频预加载完成！(共 ${preloadedVideos.size} 个episodes)`);
            updatePreloadStatus();
        }
        return;
    }
    
    // 如果已经达到最大并发数，不启动新的预加载
    if (activePreloadCount >= maxConcurrentPreload) {
        return;
    }
    
    // 从队列中取出一个episode并预加载
    const episodeId = preloadQueue.shift();
    if (!preloadedVideos.has(episodeId)) {
        preloadVideo(episodeId); // 异步执行，不等待
    } else {
        // 如果已预加载，继续下一个
        processPreloadQueue();
    }
    
    // 继续处理队列中的其他项（直到达到并发上限）
    if (preloadQueue.length > 0 && activePreloadCount < maxConcurrentPreload) {
        setTimeout(() => processPreloadQueue(), 100);
    }
}

// 启动预加载（预加载所有episodes，按顺序）
function startPreloading() {
    console.log(`🔍 startPreloading 被调用, allEpisodes.length=${allEpisodes.length}`);
    
    if (allEpisodes.length === 0) {
        console.warn('⚠️ allEpisodes 为空，无法预加载');
        return;
    }
    
    // 如果已经有预加载在进行，不重复启动
    if (preloadQueue.length > 0 || activePreloadCount > 0) {
        console.log(`⚠️ 预加载已在进行中，跳过重复启动 (队列:${preloadQueue.length}, 活跃:${activePreloadCount})`);
        return;
    }
    
    // 将所有未预加载的episodes加入队列（按顺序）
    preloadQueue = [];
    let addedCount = 0;
    
    console.log('📝 开始构建预加载队列...');
    for (const episode of allEpisodes) {
        // episode.id 而不是 episode.episode_id
        if (!preloadedVideos.has(episode.id)) {
            preloadQueue.push(episode.id);
            addedCount++;
            if (addedCount <= 3) {
                console.log(`  + 添加到队列: ${episode.id}`);
            }
        }
    }
    
    if (addedCount === 0) {
        console.log('✓ 所有episodes已预加载');
        return;
    }
    
    console.log(`🚀 开始后台预加载所有视频: ${addedCount} 个episodes (并发数: ${maxConcurrentPreload})`);
    console.log(`   队列前3个: ${preloadQueue.slice(0, 3).join(', ')}`);
    
    // 立即开始预加载（多个并发）
    for (let i = 0; i < maxConcurrentPreload && preloadQueue.length > 0; i++) {
        console.log(`   启动并发任务 ${i + 1}/${maxConcurrentPreload}`);
        setTimeout(() => processPreloadQueue(), i * 100);
    }
}

// 清理过期的预加载缓存
function cleanupPreloadCache() {
    const maxAge = 10 * 60 * 1000; // 10分钟
    const maxCount = 20; // 最多保留20个预加载的视频元素
    const now = Date.now();
    
    // 清理过期的
    for (const [episodeId, data] of preloadedVideos.entries()) {
        if (now - data.timestamp > maxAge) {
            // 释放 video 元素
            if (data.originalVideo) {
                data.originalVideo.src = '';
                data.originalVideo.load();
            }
            if (data.renderedVideo) {
                data.renderedVideo.src = '';
                data.renderedVideo.load();
            }
            preloadedVideos.delete(episodeId);
            console.log(`🗑️ 清理过期预加载: ${episodeId}`);
        }
    }
    
    // 如果数量过多，清理最旧的
    if (preloadedVideos.size > maxCount) {
        const entries = Array.from(preloadedVideos.entries())
            .sort((a, b) => a[1].timestamp - b[1].timestamp);
        
        const toDelete = entries.slice(0, preloadedVideos.size - maxCount);
        for (const [episodeId, data] of toDelete) {
            if (data.originalVideo) {
                data.originalVideo.src = '';
                data.originalVideo.load();
            }
            if (data.renderedVideo) {
                data.renderedVideo.src = '';
                data.renderedVideo.load();
            }
            preloadedVideos.delete(episodeId);
            console.log(`🗑️ 清理多余预加载: ${episodeId}`);
        }
    }
}

// 加载episodes（随机）
async function loadEpisodes(append = false) {
    const episodeList = document.getElementById('episodeList');
    const datasetPath = document.getElementById('datasetPath');
    const welcomeScreen = document.getElementById('welcomeScreen');

    if (!append && episodeList) {
        episodeList.innerHTML = '<div class="loading">正在随机加载未查看的 episodes...</div>';
    }

    if (!append && !episodeList && welcomeScreen) {
        welcomeScreen.innerHTML = '<p>正在加载 Episode...</p>';
    }
    
    try {
        const uid = getUserId();
        const response = await fetch(`/api/episodes?limit=${currentLimit}&user_id=${uid}&exclude_others=true`);
        const data = await response.json();
        
        // 更新数据集信息
        if (datasetPath) {
            const totalAcrossAll = data.datasets.reduce((sum, ds) => sum + ds.total_episodes, 0);
            const unviewedAcrossAll = data.datasets.reduce((sum, ds) => sum + ds.unviewed_episodes, 0);
            const loadedAcrossAll = data.datasets.reduce((sum, ds) => sum + ds.loaded_episodes, 0);
            let infoText = `当前显示 ${loadedAcrossAll} 个 | 未查看 ${unviewedAcrossAll}/${totalAcrossAll} (${data.datasets.length} 个数据集)`;
            datasetPath.textContent = infoText;
        }
        
        // 直接渲染所有episodes（不分组）
        if (data.episodes && data.episodes.length > 0) {
            // 保存episodes列表用于预加载
            if (!append) {
                allEpisodes = data.episodes;
                if (episodeList) {
                    episodeList.innerHTML = '';
                }
            } else {
                allEpisodes = allEpisodes.concat(data.episodes);
                // 移除旧的加载更多按钮
                if (episodeList) {
                    const oldLoadMore = episodeList.querySelector('.load-more-container');
                    if (oldLoadMore) {
                        oldLoadMore.remove();
                    }
                }
            }
            
            // 直接渲染所有episodes
            if (episodeList) {
                data.episodes.forEach(episode => {
                    const item = document.createElement('div');
                    item.className = 'episode-item';
                    item.innerHTML = `
                        <div class="episode-name">
                            <span class="dataset-badge">${episode.dataset}</span>
                            ${episode.name}
                        </div>
                        <div class="episode-meta">${episode.num_frames} 帧 | #${episode.dataset_index}</div>
                    `;
                    item.dataset.episodeId = episode.id;
                    item.dataset.episodeName = episode.name;
                    item.dataset.datasetName = episode.dataset;
                    item.dataset.episodeIndex = episode.dataset_index;
                    item.addEventListener('click', () => selectEpisode(episode.id, episode.name, episode.dataset, episode.dataset_index));
                    episodeList.appendChild(item);
                });
                
                // 检查是否还有未查看的数据
                const hasUnviewed = data.datasets.some(ds => ds.unviewed_episodes > 0);
                
                if (hasUnviewed) {
                    // 添加"获取新批次"按钮
                    const loadMoreContainer = document.createElement('div');
                    loadMoreContainer.className = 'load-more-container';
                    loadMoreContainer.innerHTML = `
                        <button class="load-more-btn refresh-btn" id="refreshBtn">
                            🔄 随机获取新的一批 (${currentLimit} 个)
                        </button>
                        <div class="hint-text">💡 从所有数据集中随机加载 ${currentLimit} 个未被任何人查看过的 episodes</div>
                    `;
                    episodeList.appendChild(loadMoreContainer);
                    
                    document.getElementById('refreshBtn').addEventListener('click', () => {
                        loadEpisodes(false);  // 不追加，而是替换
                    });
                } else {
                    // 所有都已查看
                    const noMoreContainer = document.createElement('div');
                    noMoreContainer.className = 'load-more-container';
                    noMoreContainer.innerHTML = `
                        <div class="no-more-text">🎉 所有 episodes 都已被查看过！</div>
                    `;
                    episodeList.appendChild(noMoreContainer);
                }
            }
        } else if (!append && episodeList) {
            episodeList.innerHTML = '<div class="loading">未找到任何 episode</div>';
        }
        
        // 🚀 加载完episodes后，启动后台预加载所有视频和翻译
        if (allEpisodes.length > 0) {
            console.log(`📋 已加载 ${allEpisodes.length} 个episodes，将在1秒后启动后台预加载...`);
            console.log(`首个episode: ${allEpisodes[0].id}`);
            setTimeout(() => {
                console.log('⏰ 1秒延迟结束，现在启动预加载...');
                cleanupPreloadCache(); // 清理过期缓存
                startPreloading(); // 开始预加载所有视频
                
                // 视频预加载启动后，延迟启动翻译预加载（避免同时抢占资源）
                setTimeout(() => {
                    console.log('📝 启动翻译预加载...');
                    startTranslationPreload();
                }, 3000); // 延迟3秒后启动翻译预加载
            }, 1000); // 延迟1秒，确保页面渲染完成

            if (!episodeList && !currentEpisodeId) {
                const first = allEpisodes[0];
                selectEpisode(first.id, first.name, first.dataset, first.dataset_index);
            }
        } else {
            console.warn('⚠️ allEpisodes 为空，无法启动预加载');
        }
        
    } catch (error) {
        console.error('加载episodes失败:', error);
        if (episodeList) {
            episodeList.innerHTML = '<div class="loading">加载失败，请检查配置</div>';
        }
    }
}

// 获取当前 episode 在列表中的索引
function getCurrentEpisodeIndex() {
    if (!currentEpisodeId || allEpisodes.length === 0) {
        return -1;
    }
    return allEpisodes.findIndex(ep => ep.id === currentEpisodeId);
}

// 更新导航按钮状态
function updateNavigationButtons() {
    const prevBtn = document.getElementById('prevEpisodeBtn');
    const nextBtn = document.getElementById('nextEpisodeBtn');
    
    if (!prevBtn || !nextBtn) return;
    
    const currentIndex = getCurrentEpisodeIndex();
    
    if (currentIndex === -1 || allEpisodes.length === 0) {
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        return;
    }
    
    prevBtn.disabled = currentIndex === 0;
    nextBtn.disabled = currentIndex === allEpisodes.length - 1;
}

// 导航到上一个 episode
function navigateToPrevEpisode() {
    if (hasUnsavedChanges()) {
        if (!confirm('当前 Episode 还未提交评估，确定要离开吗？')) {
            return;
        }
    }
    
    const currentIndex = getCurrentEpisodeIndex();
    if (currentIndex <= 0) return;
    
    const prevEpisode = allEpisodes[currentIndex - 1];
    selectEpisode(prevEpisode.id, prevEpisode.name, prevEpisode.dataset, prevEpisode.dataset_index);
}

// 导航到下一个 episode
function navigateToNextEpisode() {
    if (hasUnsavedChanges()) {
        if (!confirm('当前 Episode 还未提交评估，确定要离开吗？')) {
            return;
        }
    }
    
    const currentIndex = getCurrentEpisodeIndex();
    if (currentIndex === -1 || currentIndex >= allEpisodes.length - 1) return;
    
    const nextEpisode = allEpisodes[currentIndex + 1];
    selectEpisode(nextEpisode.id, nextEpisode.name, nextEpisode.dataset, nextEpisode.dataset_index);
}

// 检查是否有未保存的更改（即当前 episode 还没提交评估）
function hasUnsavedChanges() {
    // 如果当前 episode 已经提交过评估，就不算有未保存更改
    if (savedAnnotationState.submitted) {
        return false;
    }
    
    // 检查是否有任何评估内容被填写
    const checkboxes = document.querySelectorAll('input[name="issue"]:checked');
    const additionalNotes = document.getElementById('additionalNotes');
    const currentNotes = additionalNotes ? additionalNotes.value.trim() : '';
    const transErrorCheckboxes = document.querySelectorAll('.trans-error-checkbox:checked');
    
    // 如果勾选了任何问题、写了备注、或标记了翻译错误，说明有内容需要提交
    if (checkboxes.length > 0 || currentNotes !== '' || transErrorCheckboxes.length > 0) {
        return true;
    }
    
    // 什么都没填也算"未提交"——因为用户需要确认质量良好
    // 但如果是刚切过来还没看，不弹提示（通过 submitted 标记控制）
    return true; // 只要没提交过，就提示
}

// 辅助函数：比较两个数组是否相等
function arraysEqual(arr1, arr2) {
    if (arr1.length !== arr2.length) return false;
    for (let i = 0; i < arr1.length; i++) {
        if (arr1[i] !== arr2[i]) return false;
    }
    return true;
}

// 选择一个episode
async function selectEpisode(episodeId, episodeName, datasetName, episodeIndex) {
    // 如果已经有当前 episode 且未提交评估，弹提示
    if (currentEpisodeId && currentEpisodeId !== episodeId && hasUnsavedChanges()) {
        if (!confirm('当前 Episode 还未提交评估，确定要离开吗？')) {
            return;
        }
    }
    
    currentEpisodeId = episodeId;
    
    // 重置提交状态
    savedAnnotationState = {
        issues: [],
        additionalNotes: '',
        submitted: false
    };
    currentTranslations = [];
    
    // 保存episode信息用于保存标注
    window.currentEpisodeInfo = {
        id: episodeId,
        name: episodeName,
        dataset: datasetName,
        index: episodeIndex
    };
    
    // 更新选中状态（无侧边栏时可为空）
    document.querySelectorAll('.episode-item').forEach(item => {
        if (item.dataset.episodeId === episodeId) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    
    // 隐藏欢迎屏幕，显示详情
    document.getElementById('welcomeScreen').style.display = 'none';
    const detailPanel = document.getElementById('episodeDetail');
    detailPanel.style.display = 'block';
    
    // 加载episode详情
    await loadEpisodeDetail(episodeId);
    
    // 加载标注
    await loadAnnotation(episodeId);
    
    // 更新导航按钮状态
    updateNavigationButtons();
}

// 加载episode详情
async function loadEpisodeDetail(episodeId) {
    try {
        const uid = getUserId();
        const response = await fetch(`/api/episode/${episodeId}?user_id=${uid}`);
        const data = await response.json();
        
        // 更新标题
        document.getElementById('episodeTitle').textContent = `Episode: ${data.episode_name}`;
        
        // 更新信息
        const infoDiv = document.getElementById('episodeInfo');
        let infoHTML = `
            <span>📁 数据集: ${data.dataset_name}</span>
            <span>📊 总帧数: ${data.num_frames}</span>
        `;
        
        // 显示指令信息
        if (data.instructions && data.instructions.length > 0) {
            infoHTML += `<span>📝 指令数: ${data.instructions.length}</span>`;
        }
        
        infoDiv.innerHTML = infoHTML;
        
        // 如果有指令，显示在顶部
        if (data.instructions && data.instructions.length > 0) {
            displayInstructions(data.instructions);
        }
        
        // 渲染视频
        renderVideos(data.episode_id);
        
    } catch (error) {
        console.error('加载episode详情失败:', error);
        alert('加载失败，请重试');
    }
}

// 渲染视频 scrubbing 视图
function renderVideos(episodeId) {
    const framesGrid = document.getElementById('framesGrid');
    framesGrid.innerHTML = '';

    // 清理上一次的 document-level 事件监听器
    if (window._scrubCleanup) {
        window._scrubCleanup();
        window._scrubCleanup = null;
    }

    // 外层 flex 容器
    const flexRow = document.createElement('div');
    flexRow.className = 'video-side-by-side';

    // 尝试复用预加载的 video 元素
    const cached = preloadedVideos.get(episodeId);

    const originalVideo = cached ? cached.originalVideo.cloneNode(true) : document.createElement('video');
    const renderedVideo = cached ? cached.renderedVideo.cloneNode(true) : document.createElement('video');

    if (!cached) {
        originalVideo.src = `/api/episode/${episodeId}/video/original`;
        renderedVideo.src = `/api/episode/${episodeId}/video/rendered`;
    } else {
        originalVideo.src = cached.original;
        renderedVideo.src = cached.rendered;
    }

    // 通用属性
    [originalVideo, renderedVideo].forEach(v => {
        v.preload = 'auto';
        v.muted = true;
        v.playsInline = true;
        v.playbackRate = 1.0;
        v.defaultPlaybackRate = 1.0;
        v.style.pointerEvents = 'none'; // 防止视频拦截进度条拖拽事件
        v.load();
    });

    originalVideo.className = 'scrub-video scrub-video-original';
    renderedVideo.className = 'scrub-video scrub-video-rendered';

    // Original wrapper
    const originalWrapper = document.createElement('div');
    originalWrapper.className = 'video-wrapper';
    originalWrapper.id = 'originalVideoWrapper';
    const originalLabel = document.createElement('div');
    originalLabel.className = 'video-label';
    originalLabel.textContent = 'Original';
    originalWrapper.appendChild(originalLabel);
    originalWrapper.appendChild(originalVideo);

    // Rendered wrapper
    const renderedWrapper = document.createElement('div');
    renderedWrapper.className = 'video-wrapper';
    renderedWrapper.id = 'renderedVideoContainer';
    const renderedLabel = document.createElement('div');
    renderedLabel.className = 'video-label';
    renderedLabel.textContent = 'Rendered (加载中...)';
    renderedWrapper.appendChild(renderedLabel);
    renderedWrapper.appendChild(renderedVideo);

    // Rendered 视频加载状态
    renderedVideo.addEventListener('loadeddata', () => {
        renderedLabel.textContent = 'Rendered';
    });
    renderedVideo.addEventListener('error', () => {
        renderedLabel.textContent = 'Rendered (加载失败)';
        renderedLabel.style.color = '#f44336';
    });

    flexRow.appendChild(originalWrapper);
    flexRow.appendChild(renderedWrapper);

    // ========== 进度条 scrubber（基于真实秒数） ==========
    const scrubber = document.createElement('input');
    scrubber.type = 'range';
    scrubber.className = 'video-scrubber';
    scrubber.min = '0';
    scrubber.max = '100';   // 临时值，loadedmetadata 后会更新为真实 duration * 100
    scrubber.value = '0';
    scrubber.step = '1';

    // 时间显示
    const timeLabel = document.createElement('div');
    timeLabel.className = 'scrub-time-label';
    timeLabel.textContent = '0:00 / 0:00';

    // 速度切换按钮
    const speeds = [1, 2, 4, 8];
    let speedIdx = 0;
    const speedBtn = document.createElement('button');
    speedBtn.className = 'speed-btn';
    speedBtn.textContent = '1x';
    speedBtn.title = '点击切换播放速度';
    speedBtn.addEventListener('click', () => {
        speedIdx = (speedIdx + 1) % speeds.length;
        const spd = speeds[speedIdx];
        speedBtn.textContent = `${spd}x`;
        originalVideo.playbackRate = spd;
        renderedVideo.playbackRate = spd;
    });

    // 状态
    let isScrubbing = false;
    let wasPlayingBeforeScrub = false;
    let videoDuration = 0; // 真实时长（秒），由 loadedmetadata 设置

    // seek 节流状态：每帧关键帧视频 seek 很快，用简单 rAF 节流即可
    let _seekPending = false;   // 是否有待执行的 seek
    let _seekTarget = 0;        // 待 seek 的目标时间
    let _rafId = null;          // requestAnimationFrame id

    function formatTime(s) {
        const m = Math.floor(s / 60);
        const sec = Math.floor(s % 60);
        return `${m}:${sec.toString().padStart(2, '0')}`;
    }

    function updateTimeLabel(current, duration) {
        timeLabel.textContent = `${formatTime(current)} / ${formatTime(duration)}`;
    }

    // scrubber value → 秒：value 就是 centiseconds (duration * 100)
    // 这样 step=1 对应 0.01 秒精度，足够细腻
    function scrubberToTime() {
        return parseFloat(scrubber.value) / 100;
    }

    function timeToScrubber(t) {
        return Math.round(t * 100);
    }

    // 真正执行 seek 的内部函数
    function _doSeek(timeSec) {
        timeSec = Math.max(0, Math.min(timeSec, videoDuration));
        originalVideo.currentTime = Math.min(timeSec, originalVideo.duration || videoDuration);
        if (renderedVideo.duration && isFinite(renderedVideo.duration)) {
            renderedVideo.currentTime = Math.min(timeSec, renderedVideo.duration);
        }
        updateTimeLabel(timeSec, videoDuration);
    }

    // 节流 seek：拖拽时通过 rAF 合并，每帧最多 seek 一次
    function seekBothVideos(timeSec) {
        if (!isScrubbing) {
            // 非拖拽状态（如松开时的最终 seek），直接执行
            _doSeek(timeSec);
            return;
        }
        // 拖拽中：记录目标，用 rAF 节流（每帧关键帧视频不需要等 seeked）
        _seekTarget = timeSec;
        updateTimeLabel(timeSec, videoDuration); // 时间标签立即更新
        if (!_seekPending) {
            _seekPending = true;
            _rafId = requestAnimationFrame(() => {
                _seekPending = false;
                _rafId = null;
                _doSeek(_seekTarget);
            });
        }
    }

    // ---- loadedmetadata: 用真实 duration 初始化 scrubber ----
    originalVideo.addEventListener('loadedmetadata', () => {
        originalVideo.playbackRate = speeds[speedIdx];
        videoDuration = originalVideo.duration;
        scrubber.max = String(timeToScrubber(videoDuration));
        scrubber.value = '0';
        updateTimeLabel(0, videoDuration);
    });
    renderedVideo.addEventListener('loadedmetadata', () => {
        renderedVideo.playbackRate = speeds[speedIdx];
    });

    // ---- 拖拽开始：暂停视频，标记 scrubbing ----
    function startScrub() {
        isScrubbing = true;
        wasPlayingBeforeScrub = !originalVideo.paused;
        originalVideo.pause();
        renderedVideo.pause();
    }

    // ---- 拖拽中：节流 seek ----
    function onScrubInput() {
        seekBothVideos(scrubberToTime());
    }

    // ---- 拖拽结束：恢复播放 ----
    function endScrub() {
        if (!isScrubbing) return;
        isScrubbing = false;
        // 取消未执行的 rAF
        if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; _seekPending = false; }
        // 最终 seek 一次确保精确
        _doSeek(scrubberToTime());
        if (wasPlayingBeforeScrub) {
            originalVideo.play();
            renderedVideo.play();
        }
    }

    scrubber.addEventListener('mousedown', startScrub);
    scrubber.addEventListener('touchstart', startScrub, { passive: true });
    scrubber.addEventListener('input', onScrubInput);
    scrubber.addEventListener('mouseup', endScrub);
    scrubber.addEventListener('touchend', endScrub);

    // document-level mouseup 兜底（鼠标拖出 scrubber 后松开）
    function docMouseUp() { if (isScrubbing) endScrub(); }
    document.addEventListener('mouseup', docMouseUp);

    // 清理函数，下次 renderVideos 时调用
    window._scrubCleanup = () => {
        document.removeEventListener('mouseup', docMouseUp);
        if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
    };

    // ---- 播放时同步 scrubber + rendered ----
    // 关键：播放中不要每帧都 set currentTime，否则 rendered 视频会不断 seek 导致卡顿
    // 只在漂移超过阈值时才纠正
    const SYNC_DRIFT_THRESHOLD = 0.15; // 秒，超过此值才强制同步
    originalVideo.addEventListener('timeupdate', () => {
        if (isScrubbing) return; // 拖拽中不更新，避免冲突
        const t = originalVideo.currentTime;
        const dur = originalVideo.duration;
        if (!dur || !isFinite(dur)) return;
        scrubber.value = String(timeToScrubber(t));
        updateTimeLabel(t, dur);
        // 同步 rendered：仅在漂移过大时纠正，避免持续 seek 导致卡顿
        if (renderedVideo.duration && isFinite(renderedVideo.duration)) {
            const drift = Math.abs(renderedVideo.currentTime - t);
            if (drift > SYNC_DRIFT_THRESHOLD) {
                renderedVideo.currentTime = Math.min(t, renderedVideo.duration);
            }
        }
    });

    // 播放结束
    originalVideo.addEventListener('ended', () => {
        renderedVideo.pause();
    });

    // 组装
    const container = document.createElement('div');
    container.className = 'video-scrub-container';
    container.appendChild(flexRow);

    // 进度条 + 速度按钮行
    const controlsRow = document.createElement('div');
    controlsRow.className = 'scrub-controls-row';
    controlsRow.appendChild(scrubber);
    controlsRow.appendChild(speedBtn);
    container.appendChild(controlsRow);

    container.appendChild(timeLabel);
    framesGrid.appendChild(container);
}

// 显示指令信息（自动翻译，直接显示中文，带 checkbox 标记错误）
async function displayInstructions(instructions) {
    const framesSection = document.querySelector('.frames-section');
    const annotationSection = document.querySelector('.annotation-section');
    
    // 移除旧的指令显示
    const oldInstructions = document.querySelector('.instructions-display');
    if (oldInstructions) {
        oldInstructions.remove();
    }
    
    // 创建新的指令显示容器
    const instructionsDiv = document.createElement('div');
    instructionsDiv.className = 'instructions-display';
    instructionsDiv.innerHTML = `
        <div class="instructions-header">
            <h4>📝 任务指令:</h4>
            <span class="translate-status" id="translateStatus"></span>
        </div>
        <div class="instructions-list" id="instructionsList">
            <div class="loading-translations">翻译中...</div>
        </div>
    `;
    
    // 插入到视频区域和标注区域之间
    if (framesSection && annotationSection) {
        framesSection.parentNode.insertBefore(instructionsDiv, annotationSection);
    }
    
    // 尝试加载已有翻译或自动翻译
    await loadOrTranslateInstructions(instructions);
}

// 加载已有翻译或自动翻译
async function loadOrTranslateInstructions(instructions) {
    if (!currentEpisodeId) return;
    
    const listDiv = document.getElementById('instructionsList');
    const statusEl = document.getElementById('translateStatus');
    if (!listDiv) return;
    
    try {
        // 先检查数据库是否已有翻译
        const response = await fetch(`/api/instruction/${currentEpisodeId}`);
        const data = await response.json();
        
        let translations = {};
        
        if (data.success && Object.keys(data.translations).length >= instructions.length) {
            // 已有翻译，直接使用
            translations = data.translations;
            if (statusEl) statusEl.textContent = '';
        } else {
            // 没有翻译，自动调用翻译 API
            if (statusEl) statusEl.textContent = '翻译中...';
            
            const translateResponse = await fetch('/api/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ texts: instructions })
            });
            
            const translateData = await translateResponse.json();
            
            if (translateData.success && translateData.translations) {
                // 构建翻译对象
                instructions.forEach((inst, i) => {
                    translations[i] = {
                        original: inst,
                        translation: translateData.translations[i] || '[翻译失败]'
                    };
                });
                
                // 保存翻译到数据库
                const saveTranslations = instructions.map((inst, i) => ({
                    index: i,
                    original: inst,
                    translation: translateData.translations[i] || '[翻译失败]'
                }));
                
                await fetch(`/api/instruction/${currentEpisodeId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ translations: saveTranslations })
                });
                
                if (statusEl) statusEl.textContent = '';
            } else {
                throw new Error('翻译失败');
            }
        }
        
        // 渲染翻译结果（带 checkbox）
        renderTranslationItems(translations, instructions);
        
    } catch (error) {
        console.error('加载/翻译指令失败:', error);
        if (listDiv) {
            listDiv.innerHTML = '<div class="translation-error">翻译加载失败</div>';
        }
        // fallback: 显示英文原文
        currentTranslations = instructions.map((inst, i) => ({
            index: i, original: inst, translation: inst
        }));
    }
}

// 渲染翻译条目（checkbox + 中文文本）
function renderTranslationItems(translations, originalInstructions) {
    const listDiv = document.getElementById('instructionsList');
    if (!listDiv) return;
    
    listDiv.innerHTML = '';
    currentTranslations = [];
    
    const sortedKeys = Object.keys(translations).sort((a, b) => parseInt(a) - parseInt(b));
    
    sortedKeys.forEach(key => {
        const trans = translations[key];
        const translation = typeof trans === 'string' ? trans : (trans.translation || trans.text || '');
        const original = typeof trans === 'string' ? (originalInstructions ? originalInstructions[parseInt(key)] : '') : (trans.original || '');
        const idx = parseInt(key);
        
        // 保存到 currentTranslations
        currentTranslations.push({ index: idx, original: original, translation: translation });
        
        // 创建条目
        const item = document.createElement('div');
        item.className = 'trans-feedback-item';
        item.dataset.index = idx;
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'trans-error-checkbox';
        checkbox.id = `transError_${idx}`;
        checkbox.title = '勾选表示此条翻译有误';
        
        const label = document.createElement('label');
        label.className = 'trans-feedback-label';
        label.htmlFor = `transError_${idx}`;
        
        const numSpan = document.createElement('span');
        numSpan.className = 'trans-num';
        numSpan.textContent = `${idx + 1}.`;
        
        const textSpan = document.createElement('span');
        textSpan.className = 'trans-text';
        textSpan.textContent = translation;
        
        label.appendChild(numSpan);
        label.appendChild(textSpan);
        
        // 修正输入框（勾选后展开）
        const correctionDiv = document.createElement('div');
        correctionDiv.className = 'trans-correction';
        correctionDiv.style.display = 'none';
        
        const correctionInput = document.createElement('input');
        correctionInput.type = 'text';
        correctionInput.className = 'trans-correction-input';
        correctionInput.placeholder = '输入修正翻译（可选）';
        correctionInput.dataset.index = idx;
        
        correctionDiv.appendChild(correctionInput);
        
        // checkbox 切换展开修正框
        checkbox.addEventListener('change', () => {
            correctionDiv.style.display = checkbox.checked ? 'block' : 'none';
            item.classList.toggle('trans-marked-error', checkbox.checked);
            if (checkbox.checked) {
                correctionInput.focus();
            }
        });
        
        item.appendChild(checkbox);
        item.appendChild(label);
        item.appendChild(correctionDiv);
        listDiv.appendChild(item);
    });
}

// 加载标注
async function loadAnnotation(episodeId) {
    try {
        const response = await fetch(`/api/annotation/${episodeId}`);
        const data = await response.json();
        
        // 恢复勾选状态
        const issues = data.issues || [];
        const checkboxes = document.querySelectorAll('input[name="issue"]');
        checkboxes.forEach(cb => {
            cb.checked = issues.includes(cb.value);
        });
        
        // 恢复额外说明
        const additionalNotes = document.getElementById('additionalNotes');
        const notesValue = data.additional_notes || '';
        if (additionalNotes) {
            additionalNotes.value = notesValue;
        }
        
        // 判断是否已有提交过的评估（有 updated_at 或有 issues 或有 content）
        const hasExistingAnnotation = data.updated_at || issues.length > 0 || data.content;
        
        // 更新已保存状态
        savedAnnotationState = {
            issues: [...issues],
            additionalNotes: notesValue,
            submitted: !!hasExistingAnnotation
        };
        
        // 清除状态消息
        const status = document.getElementById('annotationStatus');
        if (status) {
            status.textContent = '';
        }
    } catch (error) {
        console.error('加载标注失败:', error);
    }
}

// 保存标注（同时提交质量评估 + 翻译反馈）
async function saveAnnotation() {
    if (!currentEpisodeId || !window.currentEpisodeInfo) {
        return;
    }
    
    const status = document.getElementById('annotationStatus');
    const additionalNotes = document.getElementById('additionalNotes').value.trim();
    
    // 收集勾选的问题
    const checkboxes = document.querySelectorAll('input[name="issue"]:checked');
    const issues = Array.from(checkboxes).map(cb => cb.value);
    
    // 收集翻译反馈
    const transErrorCheckboxes = document.querySelectorAll('.trans-error-checkbox:checked');
    const hasTranslationFeedback = transErrorCheckboxes.length > 0;
    
    try {
        // 1. 提交质量评估
        const response = await fetch(`/api/annotation/${currentEpisodeId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                issues: issues,
                additional_notes: additionalNotes,
                episode_name: window.currentEpisodeInfo.name,
                dataset_name: window.currentEpisodeInfo.dataset,
                episode_index: window.currentEpisodeInfo.index,
                reviewer_name: reviewerName,
                content: issues.length > 0 ? `问题: ${issues.join(', ')}` : '质量良好'
            }),
        });
        
        const data = await response.json();
        
        if (!data.ok) {
            throw new Error('保存评估失败');
        }
        
        // 2. 如果有翻译反馈，同时提交
        if (hasTranslationFeedback && currentTranslations.length > 0) {
            const translations = currentTranslations.map(t => {
                const checkbox = document.querySelector(`#transError_${t.index}`);
                const correctionInput = document.querySelector(`.trans-correction-input[data-index="${t.index}"]`);
                const isMarkedError = checkbox && checkbox.checked;
                const correction = correctionInput ? correctionInput.value.trim() : '';
                
                return {
                    index: t.index,
                    original: t.original,
                    translation: (isMarkedError && correction) ? correction : t.translation,
                    is_edited: isMarkedError
                };
            });
            
            await fetch(`/api/instruction/${currentEpisodeId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ translations })
            });
        }
        
        // 更新状态
        const isValid = data.is_valid === 1;
        const errorCount = transErrorCheckboxes.length;
        let statusText = '';
        
        if (isValid && errorCount === 0) {
            statusText = '✓ 已标记为【质量良好】';
            status.className = 'status-message success';
        } else {
            const parts = [];
            if (issues.length > 0) parts.push(`${issues.length}个问题`);
            if (errorCount > 0) parts.push(`${errorCount}条翻译有误`);
            statusText = `✓ 已标记为【需丢弃】(${parts.join(', ')})`;
            status.className = 'status-message warning';
        }
        status.textContent = statusText;
        
        // 标记为已提交
        savedAnnotationState = {
            issues: [...issues],
            additionalNotes: additionalNotes,
            submitted: true
        };
        
        // 3秒后清除消息
        setTimeout(() => {
            status.textContent = '';
            status.className = 'status-message';
        }, 3000);
        
    } catch (error) {
        console.error('保存标注失败:', error);
        status.textContent = '✗ 保存失败';
        status.className = 'status-message error';
    }
}

// ========== Instruction 预加载翻译功能 ==========

// 预加载单个 episode 的翻译（批量翻译）
async function preloadTranslation(episodeId) {
    if (preloadedTranslations.has(episodeId)) {
        return { success: true, cached: true };
    }
    
    activeTranslationCount++;
    updatePreloadStatus();
    
    try {
        // 获取 episode 数据（包含 instructions）
        const uid = getUserId();
        const response = await fetch(`/api/episode/${episodeId}?user_id=${uid}`);
        const data = await response.json();
        
        if (!data.instructions || data.instructions.length === 0) {
            // 没有 instructions，跳过
            preloadedTranslations.add(episodeId);
            return { success: true, skipped: true };
        }
        
        // 检查是否已有翻译
        const translationResponse = await fetch(`/api/instruction/${episodeId}`);
        const translationData = await translationResponse.json();
        
        if (translationData.success && Object.keys(translationData.translations).length === data.instructions.length) {
            // 已有完整翻译
            preloadedTranslations.add(episodeId);
            return { success: true, existing: true };
        }
        
        // 批量翻译所有 instructions
        const translateResponse = await fetch('/api/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texts: data.instructions })
        });
        
        const translateData = await translateResponse.json();
        
        if (translateData.success && translateData.translations) {
            // 构建翻译结果
            const translations = data.instructions.map((text, i) => ({
                index: i,
                original: text,
                translation: translateData.translations[i] || '[翻译失败]'
            }));
            
            // 保存翻译
            if (translations.length > 0) {
                await fetch(`/api/instruction/${episodeId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ translations })
                });
            }
            
            preloadedTranslations.add(episodeId);
            return { success: true, translated: translations.length };
        } else {
            throw new Error('批量翻译失败');
        }
        
    } catch (error) {
        console.warn(`翻译预加载失败 (${episodeId}):`, error.message);
        return { success: false, error: error.message };
    } finally {
        activeTranslationCount--;
        updatePreloadStatus();
        processTranslationQueue();
    }
}

// 处理翻译队列
function processTranslationQueue() {
    if (translationQueue.length === 0) {
        if (activeTranslationCount === 0 && preloadedTranslations.size > 0) {
            console.log(`🎉 所有翻译预加载完成！(共 ${preloadedTranslations.size} 个episodes)`);
        }
        return;
    }
    
    if (activeTranslationCount >= maxConcurrentTranslation) {
        return;
    }
    
    const episodeId = translationQueue.shift();
    if (!preloadedTranslations.has(episodeId)) {
        preloadTranslation(episodeId);
    } else {
        processTranslationQueue();
    }
    
    if (translationQueue.length > 0 && activeTranslationCount < maxConcurrentTranslation) {
        setTimeout(() => processTranslationQueue(), 100);
    }
}

// 启动翻译预加载
function startTranslationPreload() {
    if (allEpisodes.length === 0) {
        return;
    }
    
    if (translationQueue.length > 0 || activeTranslationCount > 0) {
        console.log('翻译预加载已在进行中');
        return;
    }
    
    // 构建翻译队列
    translationQueue = [];
    let addedCount = 0;
    
    for (const episode of allEpisodes) {
        if (!preloadedTranslations.has(episode.id)) {
            translationQueue.push(episode.id);
            addedCount++;
        }
    }
    
    if (addedCount === 0) {
        console.log('✓ 所有 episodes 的翻译已预加载');
        return;
    }
    
    console.log(`🚀 开始后台预加载翻译: ${addedCount} 个episodes (并发数: ${maxConcurrentTranslation})`);
    
    // 启动并发任务
    for (let i = 0; i < maxConcurrentTranslation && translationQueue.length > 0; i++) {
        setTimeout(() => processTranslationQueue(), i * 100);
    }
}


