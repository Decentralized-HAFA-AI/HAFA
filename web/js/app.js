const API = 'http://127.0.0.1:7476';

async function fetchJSON(endpoint) {
    try {
        const res = await fetch(API + endpoint);
        return await res.json();
    } catch (e) { return null; }
}

async function refreshData() {
    const info = await fetchJSON('/info');
    if (info) {
        document.getElementById('height').textContent = info.height;
        document.getElementById('minted').textContent = info.total_minted_hafa.toFixed(2) + ' HAFA';
        document.getElementById('reward').textContent = info.current_reward_hafa.toFixed(4) + ' HAFA';
    }

    const learn = await fetchJSON('/learning-status');
    if (learn) document.getElementById('params').textContent = learn.total_parameters.toLocaleString();

    const auto = await fetchJSON('/auto-learn/status');
    if (auto) {
        document.getElementById('buffer').textContent = auto.buffer_size;
        document.getElementById('learning').textContent = auto.is_learning ? '✅ Yes' : '❌ No';
    }

    const gpu = await fetchJSON('/gpu/info');
    if (gpu) {
        document.getElementById('backend').textContent = gpu.backend;
        document.getElementById('device').textContent = gpu.device_name;
        document.getElementById('fp16').textContent = gpu.supports_fp16 ? '✅ Yes' : '❌ No';
    }

    const fed = await fetchJSON('/federated/stats');
    if (fed) {
        document.getElementById('pool').textContent = fed.pool_size;
        document.getElementById('shared').textContent = fed.total_shared;
    }
}

async function testFederated() {
    const resultDiv = document.getElementById('action-result');
    resultDiv.style.display = 'block';
    resultDiv.textContent = 'Sending...';
    
    const res = await fetch(API + '/federated/share', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: 'HAFA Web UI is awesome!', source: 'web_ui', confidence: 0.95, peer_id: 'browser' })
    });
    const data = await res.json();
    resultDiv.textContent = JSON.stringify(data, null, 2);
    refreshData();
}

// Auto-refresh every 5 seconds
setInterval(refreshData, 5000);
refreshData();
