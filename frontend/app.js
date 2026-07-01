const API_URL = "http://127.0.0.1:8000";

// --- VIEW NAVIGATION ---
function showView(viewId) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
    
    document.getElementById(viewId).classList.add('active');
    
    if(viewId === 'dashboard-view') {
        document.getElementById('nav-dashboard').classList.add('active');
        fetchDashboardStats();
    } else if(viewId === 'scan-view') {
        document.getElementById('nav-scan').classList.add('active');
        startCamera();
    } else if(viewId === 'profiles-view') {
        document.getElementById('nav-profiles').classList.add('active');
        fetchProfiles();
    }
}

// --- MODAL LOGIC ---
function showModal(modalId) { document.getElementById(modalId).classList.add('show'); }
function closeModal(modalId) { document.getElementById(modalId).classList.remove('show'); }

// --- API FETCHERS ---
async function fetchDashboardStats() {
    try {
        let res = await fetch(`${API_URL}/stats/today`);
        let data = await res.json();
        document.getElementById('total-entries').textContent = data.total_entries_today || 0;
    } catch (e) { console.error("Error fetching stats", e); }
}

async function fetchProfiles() {
    try {
        let res = await fetch(`${API_URL}/users`);
        let users = await res.json();
        const tbody = document.getElementById('profiles-tbody');
        tbody.innerHTML = '';
        
        users.forEach(user => {
            let name = user.french_name || user.arabic_name || "Unknown";
            let tr = document.createElement('tr');
            tr.className = 'clickable-row'; // UX enhancement
            tr.onclick = () => loadProfileDetail(user.nin);
            
            tr.innerHTML = `
                <td>${user.nin}</td>
                <td><strong>${name}</strong></td>
                <td><span class="badge">${user.category}</span></td>
                <td>${new Date(user.created_at).toLocaleDateString()}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) { console.error("Error fetching profiles", e); }
}

let currentLoadedNIN = null;

async function loadProfileDetail(nin) {
    showView('profile-detail-view');
    currentLoadedNIN = nin;
    
    try {
        // Fetch User Info
        let userRes = await fetch(`${API_URL}/users/${nin}`);
        let user = await userRes.json();
        
        document.getElementById('pd-nin').value = user.nin;
        document.getElementById('pd-french').value = user.french_name || "";
        document.getElementById('pd-arabic').value = user.arabic_name || "";
        document.getElementById('pd-category').value = user.category;
        
        // Fetch Logs
        let logsRes = await fetch(`${API_URL}/users/${nin}/logs`);
        let logs = await logsRes.json();
        
        const tbody = document.getElementById('pd-logs-tbody');
        tbody.innerHTML = '';
        
        if (logs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="2">No logs found.</td></tr>';
            return;
        }
        
        logs.forEach(log => {
            let actionClass = log.action === 'CHECK_IN' ? 'log-checkin' : 'log-checkout';
            let tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="${actionClass}">${log.action}</td>
                <td>${new Date(log.timestamp).toLocaleString()}</td>
                <td><button class="btn-icon" onclick="deleteLog(${log.id}, '${nin}')"><i class="fas fa-trash"></i></button></td>
            `;
            tbody.appendChild(tr);
        });
    } catch(e) { console.error(e); }
}

// --- PROFILE EDIT & DELETE LOGIC ---
document.getElementById('edit-profile-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const newNin = document.getElementById('pd-nin').value;
    const payload = {
        old_nin: currentLoadedNIN,
        nin: newNin,
        french_name: document.getElementById('pd-french').value,
        arabic_name: document.getElementById('pd-arabic').value,
        category: document.getElementById('pd-category').value
    };
    
    try {
        let res = await fetch(`${API_URL}/users/${currentLoadedNIN}`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        if (res.ok) {
            alert("Profile Updated Successfully!");
            currentLoadedNIN = newNin; // update state
            fetchProfiles(); // refresh list in background
        }
    } catch(e) { console.error(e); }
});

async function deleteProfile() {
    if (!confirm("Are you sure you want to delete this profile and ALL its logs?")) return;
    try {
        let res = await fetch(`${API_URL}/users/${currentLoadedNIN}`, { method: 'DELETE' });
        if (res.ok) {
            alert("Profile Deleted");
            showView('profiles-view');
        }
    } catch(e) { console.error(e); }
}

async function deleteLog(logId, nin) {
    if (!confirm("Are you sure you want to delete this log?")) return;
    try {
        let res = await fetch(`${API_URL}/logs/${logId}`, { method: 'DELETE' });
        if (res.ok) {
            loadProfileDetail(nin); // reload logs
        }
    } catch(e) { console.error(e); }
}

// --- SCANNER & CAMERA LOGIC ---
const video = document.getElementById('videoElement');
const ipVideo = document.getElementById('ipVideoElement');
const canvas = document.getElementById('canvasElement');
const extractBtn = document.getElementById('extractBtn');
const loadingIndicator = document.getElementById('loadingIndicator');

let useIpCamera = false;
let currentScannedNIN = null;

// Two-sided scan state
let scanState = "IDLE"; // IDLE, WAITING_FOR_BACK
let pendingData = {};

function resetScanState() {
    scanState = "IDLE";
    pendingData = {};
    document.getElementById('scan-status-banner').textContent = "Ready to Scan";
    document.getElementById('scan-status-banner').style.backgroundColor = "var(--primary-color)";
    document.getElementById('resetScanBtn').classList.add('hidden');
}

function toggleCameraSource() {
    useIpCamera = document.getElementById('cam-toggle').checked;
    
    if (useIpCamera) {
        document.getElementById('ip-webcam-settings').classList.remove('hidden');
        video.style.display = 'none';
        ipVideo.style.display = 'block';
        connectIPWebcam();
        
        // Stop device camera
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
    } else {
        document.getElementById('ip-webcam-settings').classList.add('hidden');
        video.style.display = 'block';
        ipVideo.style.display = 'none';
        ipVideo.src = "";
        startCamera();
    }
}

function connectIPWebcam() {
    if (!useIpCamera) return;
    const url = document.getElementById('ip-cam-url').value;
    if (url) ipVideo.src = url;
}

async function startCamera() {
    if (useIpCamera) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        video.srcObject = stream;
    } catch (err) {
        console.error("Error accessing camera:", err);
    }
}

// Capture frame and send to API
extractBtn.addEventListener('click', async () => {
    let sourceElement = useIpCamera ? ipVideo : video;
    
    if (useIpCamera) {
        canvas.width = ipVideo.naturalWidth;
        canvas.height = ipVideo.naturalHeight;
    } else {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }
    
    canvas.getContext('2d').drawImage(sourceElement, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');
        
        if (scanState === "WAITING_FOR_BACK") {
            formData.append('expected_type', 'ID CARD VERSO');
        }

        extractBtn.disabled = true;
        loadingIndicator.classList.remove('hidden');

        try {
            const response = await fetch(`${API_URL}/extract`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error("OCR Failed");
            
            const data = await response.json();
            console.log("OCR Result:", data);
            
            let docType = data.document_type || "UNKNOWN";
            
            // Handle missing_recto error from backend
            if (data.error && data.status === "missing_recto") {
                alert(data.error);
                resetScanState();
                return;
            }

            // If waiting for back but scanned something completely invalid
            if (scanState === "WAITING_FOR_BACK" && docType !== "ID CARD VERSO" && docType !== "id_card") {
                alert("Please scan the BACK (Verso) of the card.");
                return;
            }
            
            // If idle but scanned back
            if (scanState === "IDLE" && docType === "ID CARD VERSO") {
                alert("Please scan the FRONT (Recto) of the ID card first.");
                return;
            }

            if (scanState === "IDLE") {
                if (!data.nin || data.nin === "Not found") {
                    alert("NIN could not be read clearly. Please try again.");
                    return;
                }
                
                if (docType === "ID CARD RECTO" || docType === "ID CARD RECTO (OLD)") {
                    // Start two-sided process
                    pendingData = { nin: data.nin, arabic_name: data.arabic_name };
                    scanState = "WAITING_FOR_BACK";
                    document.getElementById('scan-status-banner').textContent = "Front scanned successfully. Please flip the card and scan the BACK.";
                    document.getElementById('scan-status-banner').style.backgroundColor = "var(--success-color)";
                    document.getElementById('resetScanBtn').classList.remove('hidden');
                    return; // Stop here and wait for next scan
                }
                
                // Driver License (One-sided)
                pendingData = { nin: data.nin, arabic_name: data.arabic_name, french_name: data.french_name };
            } else if (scanState === "WAITING_FOR_BACK") {
                // Combine data
                pendingData.french_name = data.french_name;
            }

            // Proceed with full data
            currentScannedNIN = pendingData.nin;
            let finalNin = pendingData.nin;
            let finalFrench = pendingData.french_name;
            let finalArabic = pendingData.arabic_name;
            
            if (scanState === "WAITING_FOR_BACK") {
                resetScanState(); // Reset UI after we extracted our data
            }

            const userRes = await fetch(`${API_URL}/users/${finalNin}`);
            
            if (userRes.status === 404) {
                // WORKFLOW A
                document.getElementById('cp-nin').value = finalNin;
                document.getElementById('cp-french').value = finalFrench && finalFrench !== "Not found" ? finalFrench : "";
                document.getElementById('cp-arabic').value = finalArabic && finalArabic !== "Not found" ? finalArabic : "";
                showModal('modal-create');
            } else if (userRes.ok) {
                // WORKFLOW B
                const user = await userRes.json();
                document.getElementById('mc-name').textContent = user.french_name || user.arabic_name || "Unknown";
                document.getElementById('mc-nin').textContent = user.nin;
                
                const btnIn = document.getElementById('btn-checkin');
                const btnOut = document.getElementById('btn-checkout');
                
                if (user.last_action === 'CHECK_IN') {
                    btnIn.disabled = true;
                    btnOut.disabled = false;
                } else {
                    btnIn.disabled = false;
                    btnOut.disabled = true;
                }
                
                showModal('modal-check');
            }
        } catch (error) {
            console.error("Error:", error);
            alert("An error occurred during extraction.");
        } finally {
            extractBtn.disabled = false;
            loadingIndicator.classList.add('hidden');
        }
    }, 'image/jpeg');
});

// --- SUBMIT WORKFLOW A ---
document.getElementById('create-profile-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
        nin: document.getElementById('cp-nin').value,
        french_name: document.getElementById('cp-french').value,
        arabic_name: document.getElementById('cp-arabic').value,
        category: document.getElementById('cp-category').value
    };
    
    try {
        await fetch(`${API_URL}/users`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        await logAction(payload.nin, 'CHECK_IN');
        closeModal('modal-create');
        alert("Profile Created and Checked In!");
    } catch(e) { console.error(e); }
});

// --- SUBMIT WORKFLOW B ---
document.getElementById('btn-checkin').addEventListener('click', async () => {
    await logAction(currentScannedNIN, 'CHECK_IN');
    closeModal('modal-check');
    alert("Checked In Successfully!");
});

document.getElementById('btn-checkout').addEventListener('click', async () => {
    await logAction(currentScannedNIN, 'CHECK_OUT');
    closeModal('modal-check');
    alert("Checked Out Successfully!");
});

async function logAction(nin, action) {
    try {
        await fetch(`${API_URL}/users/${nin}/log`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({action: action})
        });
    } catch(e) { console.error("Log error", e); }
}

// Init
fetchDashboardStats();
