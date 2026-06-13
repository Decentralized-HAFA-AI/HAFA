// Update Dashboard
async function updateDashboard() {
    console.log('🔄 Updating dashboard...');

    // Blockchain Info
    const blockchain = await getBlockchainInfo();
    if (blockchain) {
        document.getElementById('blockchain-height').textContent = blockchain.height;
        document.getElementById('total-minted').textContent = 
            `${(blockchain.total_minted_hafa).toFixed(2)} HAFA`;
        document.getElementById('current-reward').textContent = 
            `${(blockchain.current_reward_hafa).toFixed(4)} HAFA`;
    }

    // Learning Status
    const learning = await getLearningStatus();
    if (learning) {
        document.getElementById('total-parameters').textContent = 
            learning.total_parameters.toLocaleString();
    }

    // Auto-Learn Status
    const autoLearn = await getAutoLearnStatus();
    if (autoLearn) {
        document.getElementById('buffer-size').textContent = autoLearn.buffer_size;
        document.getElementById('is-learning').textContent = 
            autoLearn.is_learning ? '✅ Yes' : '❌ No';
    }

    // GPU Info
    const gpu = await getGpuInfo();
    if (gpu) {
        document.getElementById('gpu-backend').textContent = gpu.backend;
        document.getElementById('gpu-device').textContent = gpu.device_name;
        document.getElementById('gpu-fp16').textContent = 
            gpu.supports_fp16 ? '✅ Yes' : '❌ No';
    }

    // P2P Info
    const p2p = await getP2PInfo();
    if (p2p) {
        document.getElementById('peer-id').textContent = 
            p2p.peer_id.substring(0, 20) + '...';
        document.getElementById('p2p-running').textContent = 
            p2p.is_running ? '✅ Yes' : '❌ No';
    }

    // Federated Stats
    const federated = await getFederatedStats();
    if (federated) {
        document.getElementById('pool-size').textContent = federated.pool_size;
        document.getElementById('total-shared').textContent = federated.total_shared;
    }

    // Knowledge Graph Stats
    const knowledge = await getKnowledgeStats();
    if (knowledge) {
        document.getElementById('kg-entities').textContent = knowledge.total_entities;
        document.getElementById('kg-relations').textContent = knowledge.total_relations;
    }

    console.log('✅ Dashboard updated!');
}

// Refresh All
function refreshAll() {
    updateDashboard();
}

// Auto-refresh every 10 seconds
setInterval(updateDashboard, 10000);

// Initial load
updateDashboard();