// API Configuration
const API_BASE = 'http://127.0.0.1:7476';

// API Helper Functions
async function apiGet(endpoint) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`);
        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        return null;
    }
}

async function apiPost(endpoint, data) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        return null;
    }
}

// Data Fetching Functions
async function getBlockchainInfo() {
    return await apiGet('/info');
}

async function getLearningStatus() {
    return await apiGet('/learning-status');
}

async function getAutoLearnStatus() {
    return await apiGet('/auto-learn/status');
}

async function getGpuInfo() {
    return await apiGet('/gpu/info');
}

async function getP2PInfo() {
    return await apiGet('/p2p/info');
}

async function getFederatedStats() {
    return await apiGet('/federated/stats');
}

async function getKnowledgeStats() {
    return await apiGet('/knowledge/stats');
}