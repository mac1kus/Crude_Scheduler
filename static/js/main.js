/**
 * Refinery Crude Oil Scheduling System - ENHANCED
 * All JavaScript consolidated in main.js (moved from HTML)
 */

// Global variables
let currentResults = null;

// Configuration objects
const ALERT_TYPES = {
    SUCCESS: 'success',
    WARNING: 'warning',
    DANGER: 'danger',
    INFO: 'info'
};

const TANK_STATUS_COLORS = {
    READY: '#28a745',
    FEEDING: '#28a745',
    SETTLING: '#ffc107',
    LAB_TESTING: '#ffd700',
    FILLING: '#007bff',
    FILLED: '#007bff',
    EMPTY: '#6c757d'
};

const API_ENDPOINTS = {
    SIMULATE: '/api/simulate',
    BUFFER_ANALYSIS: '/api/buffer_analysis',
    CARGO_OPTIMIZATION: '/api/cargo_optimization',
    SAVE_INPUTS: '/api/save_inputs',
    LOAD_INPUTS: '/api/load_inputs',
    EXPORT_DATA: '/api/export_data',
    EXPORT_TANK_STATUS: '/api/export_tank_status'
};

/**
 * Utility Functions
 */
const Utils = {
    formatNumber: (num) => Math.round(num).toLocaleString(),

    showLoading: (show = true) => {
        const loading = document.getElementById('loading');
        if (loading) loading.style.display = show ? 'block' : 'none';
        document.querySelectorAll('button').forEach(btn => btn.disabled = show);
    },

    showResults: () => {
        const results = document.getElementById('results');
        if (results) results.style.display = 'block';
    },

    getTankLevelColor: (volume, deadBottom) => {
        if (volume <= deadBottom) return '#dc3545';
        if (volume < deadBottom * 3) return '#ffc107';
        return '#28a745';
    },

    getStatusColor: (status) => TANK_STATUS_COLORS[status] || '#000',

    createAlert: (type, message) =>
        `<div class="alert alert-${type}">${message}</div>`,

    createMetricCard: (title, value, label, extraContent = '') => `
        <div class="metric-card">
            <h4>${title}</h4>
            <div class="metric-value">${value}</div>
            <div class="metric-label">${label}</div>
            ${extraContent}
        </div>
    `
};

// ===== MOVED FROM HTML =====
// Navigation functions
function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}


function scrollToBottom() {
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

function scrollToSimulation() {
    const element = document.querySelector('.btn-group');
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

// Tank management functions (moved from HTML, using HTML versions)
// ===== CRUDE MIX TABLE FUNCTIONS =====

function addCrudeRow() {
    const tableBody = document.getElementById('crudeMixTableBody');
    const newRow = document.createElement('tr');
    newRow.className = 'crude-mix-row';
    newRow.innerHTML = `
        <td style="padding: 5px;"><input type="text" class="crude-name-input" value="" placeholder="Enter crude name" style="width: 90%;" onchange="updateCrudeMix()"></td>
        <td style="padding: 5px; text-align: center;"><input type="number" class="crude-percentage-input" value="0" min="0" max="100" step="0.1" style="width: 80px;" onchange="updateCrudeMix()"></td>
        <td style="padding: 5px; text-align: center;"><span class="crude-volume-display">0</span></td>
        <td style="padding: 5px; text-align: center;"><button class="remove-crude-btn" onclick="removeCrudeRow(this)" style="background-color: #dc3545; color: white; border: none; padding: 3px 8px; cursor: pointer;">✕</button></td>
    `;
    tableBody.appendChild(newRow);
    updateCrudeMix(); // Recalculate totals
}

function removeCrudeRow(button) {
    // Find the parent row (tr) of the button and remove it
    button.closest('tr').remove();
    updateCrudeMix(); // Recalculate totals
}

function updateCrudeMix() {
    const processingRate = parseFloat(document.getElementById('processingRate').value) || 0;
    const rows = document.querySelectorAll('.crude-mix-row');
    let totalPercentage = 0;

    rows.forEach(row => {
        const percentageInput = row.querySelector('.crude-percentage-input');
        const percentage = parseFloat(percentageInput.value) || 0;
        totalPercentage += percentage;

        const volumeDisplay = row.querySelector('.crude-volume-display');
        const dailyVolume = (processingRate * percentage) / 100;
        volumeDisplay.textContent = dailyVolume.toLocaleString(undefined, { maximumFractionDigits: 0 });
    });

    // Update the footer totals
    document.getElementById('totalPercentage').textContent = totalPercentage.toFixed(1);
    document.getElementById('totalVolume').textContent = ((processingRate * totalPercentage) / 100).toLocaleString(undefined, { maximumFractionDigits: 0 });

    // Show or hide the 100% warning
    const warningDiv = document.getElementById('crudeMixWarning');
    const warningPercentageSpan = document.getElementById('warningPercentage');
    if (Math.abs(totalPercentage - 100) > 0.01) {
        warningPercentageSpan.textContent = totalPercentage.toFixed(1);
        warningDiv.style.display = 'block';
    } else {
        warningDiv.style.display = 'none';
    }
    autoSaveInputs(); // Save changes
}

function resetCrudeMix() {
    const tableBody = document.getElementById('crudeMixTableBody');
    // Clear existing rows
    tableBody.innerHTML = ''; 
    // Add default rows back
    tableBody.innerHTML = `
        <tr class="crude-mix-row">
            <td style="padding: 5px;"><input type="text" class="crude-name-input" value="Arab Light" placeholder="Enter crude name" style="width: 90%;" onchange="updateCrudeMix()"></td>
            <td style="padding: 5px; text-align: center;"><input type="number" class="crude-percentage-input" value="50" min="0" max="100" step="0.1" style="width: 80px;" onchange="updateCrudeMix()"></td>
            <td style="padding: 5px; text-align: center;"><span class="crude-volume-display">20,000</span></td>
            <td style="padding: 5px; text-align: center;"><button class="remove-crude-btn" onclick="removeCrudeRow(this)" style="background-color: #dc3545; color: white; border: none; padding: 3px 8px; cursor: pointer;">✕</button></td>
        </tr>
        <tr class="crude-mix-row">
            <td style="padding: 5px;"><input type="text" class="crude-name-input" value="Arab Heavy" placeholder="Enter crude name" style="width: 90%;" onchange="updateCrudeMix()"></td>
            <td style="padding: 5px; text-align: center;"><input type="number" class="crude-percentage-input" value="30" min="0" max="100" step="0.1" style="width: 80px;" onchange="updateCrudeMix()"></td>
            <td style="padding: 5px; text-align: center;"><span class="crude-volume-display">15,000</span></td>
            <td style="padding: 5px; text-align: center;"><button class="remove-crude-btn" onclick="removeCrudeRow(this)" style="background-color: #dc3545; color: white; border: none; padding: 3px 8px; cursor: pointer;">✕</button></td>
        </tr>
        <tr class="crude-mix-row">
            <td style="padding: 5px;"><input type="text" class="crude-name-input" value="Murban" placeholder="Enter crude name" style="width: 90%;" onchange="updateCrudeMix()"></td>
            <td style="padding: 5px; text-align: center;"><input type="number" class="crude-percentage-input" value="10" min="0" max="100" step="0.1" style="width: 80px;" onchange="updateCrudeMix()"></td>
            <td style="padding: 5px; text-align: center;"><span class="crude-volume-display">15,000</span></td>
            <td style="padding: 5px; text-align: center;"><button class="remove-crude-btn" onclick="removeCrudeRow(this)" style="background-color: #dc3545; color: white; border: none; padding: 3px 8px; cursor: pointer;">✕</button></td>
        </tr>
        <tr class="crude-mix-row">
            <td style="padding: 5px;"><input type="text" class="crude-name-input" value="Erha" placeholder="Enter crude name" style="width: 90%;" onchange="updateCrudeMix()"></td>
            <td style="padding: 5px; text-align: center;"><input type="number" class="crude-percentage-input" value="10" min="0" max="100" step="0.1" style="width: 80px;" onchange="updateCrudeMix()"></td>
            <td style="padding: 5px; text-align: center;"><span class="crude-volume-display">0</span></td>
            <td style="padding: 5px; text-align: center;"><button class="remove-crude-btn" onclick="removeCrudeRow(this)" style="background-color: #dc3545; color: white; border: none; padding: 3px 8px; cursor: pointer;">✕</button></td>
        </tr>
    `;
    updateCrudeMix(); // Recalculate totals
}

function updateTankCapacities() {
    const tankCapacity = document.getElementById('tankCapacity').value;
    const numTanks = getCurrentTankCount();
    
    if (tankCapacity && parseFloat(tankCapacity) > 0) {
        // Update max attribute for all existing tank level inputs
        for (let i = 1; i <= numTanks; i++) {
            const tankLevelInput = document.getElementById(`tank${i}Level`);
            if (tankLevelInput) {
                tankLevelInput.setAttribute('max', tankCapacity);
            }
        }
        // Save the change
        autoSaveInputs();
    }
}

function updateTankCount() {
    const numTanks = parseInt(document.getElementById('numTanks').value);
    const tankCountDisplay = document.getElementById('tankCountDisplay');
    tankCountDisplay.textContent = `tanks (${numTanks} tanks total)`;
    
    // Update tank grid to show/hide tanks based on count
    const tankGrid = document.getElementById('tankGrid');
    const existingTanks = tankGrid.querySelectorAll('.tank-box').length;
    
    if (numTanks > existingTanks) {
        // Add new tanks
        for (let i = existingTanks + 1; i <= numTanks; i++) {
            addNewTankBox(i);
        }
    } else if (numTanks < existingTanks) {
        // Remove extra tanks
        for (let i = existingTanks; i > numTanks; i--) {
            const tankBox = tankGrid.querySelector(`.tank-box:nth-child(${i})`);
            if (tankBox) {
                tankBox.remove();
            }
        }
    }
}

function addOneTank() {
    const numTanksInput = document.getElementById('numTanks');
    const currentCount = parseInt(numTanksInput.value);
    numTanksInput.value = currentCount + 1;
    updateTankCount();
    autoSaveInputs();
}

function removeOneTank() {
    const numTanksInput = document.getElementById('numTanks');
    const currentCount = parseInt(numTanksInput.value);
    
    // Set a minimum limit to prevent removing all tanks
    const minTanks = 1;

    if (currentCount > minTanks) {
        // Decrease the value of the number input field
        numTanksInput.value = currentCount - 1;

        // Find the last tank's HTML element using its index and remove it
        const tankGrid = document.getElementById('tankGrid');
        const lastTankBox = tankGrid.querySelector(`.tank-box:nth-child(${currentCount})`);
        
        if (lastTankBox) {
            tankGrid.removeChild(lastTankBox);
        }

        // Update the displayed tank count and save the inputs
        updateTankCount();
        autoSaveInputs();
    }
}

function addNewTankBox(tankNumber) {
    const tankGrid = document.getElementById('tankGrid');
    const tankCapacity = document.getElementById('tankCapacity').value;
    
    const tankBox = document.createElement('div');
    tankBox.className = 'tank-box';
    tankBox.innerHTML = `
        <h4>Tank ${tankNumber}</h4>
        <div class="tank-input-row">
            <label>Current Level:</label>
            <input type="number" id="tank${tankNumber}Level" value="0" min="0" max="${tankCapacity}" onchange="autoSaveInputs()">
            <span>bbl</span>
        </div>
        <div class="tank-input-row">
            <label>Dead Bottom:</label>
            <input type="number" id="deadBottom${tankNumber}" value="10000" min="10000" max="10500" onchange="autoSaveInputs()">
            <span>bbl</span>
        </div>
    `;
    
    tankGrid.appendChild(tankBox);
}

/**
 * Get current tank count dynamically
 */
function getCurrentTankCount() {
    const count = parseInt(document.getElementById('numTanks').value);
    // Return the parsed number if it's a valid non-negative integer, otherwise return 0.
    return !isNaN(count) && count >= 0 ? count : 0;
}

/**
 * AUTO-POPULATE TANK LEVELS FROM TANK CAPACITY - Updated for dynamic tanks
 */
function populateTankLevels() {
    const tankCapacity = document.getElementById('tankCapacity').value;
    const numTanks = getCurrentTankCount();

    if (tankCapacity && parseFloat(tankCapacity) > 0) {
        // Populate all active tank levels with tank capacity
        for (let i = 1; i <= numTanks; i++) {
            const tankLevelInput = document.getElementById(`tank${i}Level`);
            if (tankLevelInput) {
                tankLevelInput.value = tankCapacity;
                // FIXED: Also update the max attribute
                tankLevelInput.setAttribute('max', tankCapacity);
            }
        }
        console.log(`All ${numTanks} tanks populated with ${parseFloat(tankCapacity).toLocaleString()} bbl`);
        validateInventoryRange();
        // FIXED: Save after updating tank capacities
        autoSaveInputs();
    }
}

/**
 * TOGGLE DEPARTURE MODE
 */
function toggleDepartureMode() {
    const mode = document.getElementById('departureMode').value;
    const manualSection = document.getElementById('manualDepartureSection');
    const solverSection = document.getElementById('solverDepartureSection');

    if (mode === 'manual') {
        manualSection.style.display = 'block';
        solverSection.style.display = 'none';
    } else {
        manualSection.style.display = 'none';
        solverSection.style.display = 'block';
    }
}

/**
 * APPLY DEFAULT DEAD BOTTOM - Updated for dynamic tanks
 */
function applyDefaultDeadBottom() {
    const defaultValue = document.getElementById('defaultDeadBottom').value;
    const actualTankCount = document.querySelectorAll('.tank-box').length;
    
    for (let i = 1; i <= actualTankCount; i++) {
        const deadBottomInput = document.getElementById(`deadBottom${i}`);
        if (deadBottomInput) {
            deadBottomInput.value = defaultValue;
        }
    }
    autoSaveInputs();
}

/**
 * COLLECT FORM DATA - Improved for dynamic tanks
 */
function collectCrudeMixData() {
    const crudeData = [];
    const rows = document.querySelectorAll('.crude-mix-row');
    
    rows.forEach(row => {
        const nameInput = row.querySelector('.crude-name-input');
        const percentageInput = row.querySelector('.crude-percentage-input');
        
        if (nameInput && percentageInput) {
            crudeData.push({
                name: nameInput.value || '',
                percentage: parseFloat(percentageInput.value) || 0
            });
        }
    });
    
    return crudeData;
}
function recreateCrudeMixTable(crudeMixData) {
    if (!crudeMixData || crudeMixData.length === 0) {
        return; // Keep existing table if no saved data
    }

    const tableBody = document.getElementById('crudeMixTableBody');
    tableBody.innerHTML = ''; // Clear existing rows

    crudeMixData.forEach(crude => {
        const newRow = document.createElement('tr');
        newRow.className = 'crude-mix-row';
        newRow.innerHTML = `
            <td style="padding: 5px;"><input type="text" class="crude-name-input" value="${crude.name}" placeholder="Enter crude name" style="width: 90%;" onchange="updateCrudeMix()"></td>
            <td style="padding: 5px; text-align: center;"><input type="number" class="crude-percentage-input" value="${crude.percentage}" min="0" max="100" step="0.1" style="width: 80px;" onchange="updateCrudeMix()"></td>
            <td style="padding: 5px; text-align: center;"><span class="crude-volume-display">0</span></td>
            <td style="padding: 5px; text-align: center;"><button class="remove-crude-btn" onclick="removeCrudeRow(this)" style="background-color: #dc3545; color: white; border: none; padding: 3px 8px; cursor: pointer;">✕</button></td>
        `;
        tableBody.appendChild(newRow);
    });

    updateCrudeMix(); // Recalculate totals
}


function collectFormData() {
    const data = {};

    // Collect all input values more reliably
    document.querySelectorAll('input, select, textarea').forEach(input => {
        if (input.id && input.id !== '') {
            if (input.type === 'checkbox') {
                data[input.id] = input.checked;
            } else if (input.type === 'radio') {
                if (input.checked) {
                    data[input.id] = input.value;
                }
            } else if (input.type === 'number') {
                data[input.id] = parseFloat(input.value) || 0;
            } else {
                data[input.id] = input.value || '';
            }
        }
    });

    // --- ADD THIS CALCULATION ---
    const days = parseFloat(data.horizonDays) || 0;
    const hours = parseFloat(data.horizonHours) || 0;
    const minutes = parseFloat(data.horizonMinutes) || 0;

    // Calculate total days for the backend (e.g., 30 days, 12 hours = 30.5 days)
    data.schedulingWindow = days + (hours / 24) + (minutes / (24 * 60)); 
    // The name 'schedulingWindow' is kept for backward compatibility with the backend API

    // FIXED: Add crude mix data specifically
    data.crudeMixData = collectCrudeMixData();  
   
    return data;
}
/**
 * AUTO-SAVE INPUTS - Improved
 */
async function autoSaveInputs() {
    try {
        const inputs = collectFormData();
        
        // Save to localStorage immediately
        localStorage.setItem('refineryInputs', JSON.stringify(inputs));
        console.log('Inputs saved to localStorage');
        
        // Try to save to backend (don't block if it fails)
        try {
            const response = await fetch(API_ENDPOINTS.SAVE_INPUTS, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(inputs)
            });
            
            if (response.ok) {
                console.log('Inputs saved to server');
                // Optional: Show a brief success indicator
                showSaveStatus('saved');
            } else {
                console.log('Server save failed, but localStorage saved');
            }
        } catch (serverError) {
            console.log('Server unavailable, but localStorage saved');
        }
        
    } catch (e) {
        console.error('Save error:', e);
    }
}

/**
 * AUTO-LOAD INPUTS - Improved
 */
async function autoLoadInputs() {
    try {
        // Try localStorage first (faster)
        const saved = localStorage.getItem('refineryInputs');
        if (saved) {
            const savedInputs = JSON.parse(saved);
            applyInputValues(savedInputs);
            console.log('Inputs loaded from localStorage');
        }
        
        // Then try server (will override localStorage if successful)
        try {
            const response = await fetch(API_ENDPOINTS.LOAD_INPUTS);
            if (response.ok) {
                const serverInputs = await response.json();
                if (Object.keys(serverInputs).length > 0) {
                    applyInputValues(serverInputs);
                    console.log('Inputs loaded from server');
                }
            }
        } catch (serverError) {
            console.log('Server load failed, using localStorage');
        }
        
    } catch (e) {
        console.log('Load error:', e);
    }
}

/**
 * RUN SIMULATION
 */
/**
 * RUN SIMULATION - CORRECTED VERSION
 * This version checks if an optimized schedule exists in 'currentResults' and uses it.
 */
async function runSimulation() {
    try {
        Utils.showLoading(true);

        // --- 1. Parameter Collection and Preparation ---
        let params;
        if (currentResults && currentResults.parameters && currentResults.parameters.optimized_cargo_schedule) {
            console.log("Running simulation with the OPTIMIZED cargo schedule.");
            params = currentResults.parameters; 
            params.cargo_schedule = params.optimized_cargo_schedule;
            const freshData = collectFormData();
            // Update time and gap parameters with fresh UI values
            params.berth_gap_hours_min = freshData.berth_gap_hours_min;
            params.berth_gap_hours_max = freshData.berth_gap_hours_max;
            params.horizonDays = freshData.horizonDays;
            params.horizonHours = freshData.horizonHours;
            params.horizonMinutes = freshData.horizonMinutes;
            params.schedulingWindow = freshData.schedulingWindow;
        } else {
            console.log("Running a standard simulation from UI data.");
            params = collectFormData();
        }

        if (typeof params.schedulingWindow !== 'number' || isNaN(params.schedulingWindow)) {
             throw new Error('Simulation Horizon (schedulingWindow) is missing or invalid.');
        }

        // --- 2. API Call to Backend ---
        const response = await fetch(API_ENDPOINTS.SIMULATE, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        
        if (!response.ok) {
            throw new Error('Simulation request failed');
        }

        // --- 3. Process Results ---
        const simulationData = await response.json();
        if (simulationData.error) {
            alert('Simulation Error: ' + simulationData.error);
            return;
        }

        // Transform and store results (Keeping existing logic for data processing)
        if (simulationData.simulation_data) {
            simulationData.simulation_data = simulationData.simulation_data.map((row, index) => {
                const dateStr = row["Date"] || "";
                const openingStock = parseFloat((row["Opening Stock (bbl)"] || "0").replace(/,/g, ''));
                const closingStock = parseFloat((row["Closing Stock (bbl)"] || "0").replace(/,/g, ''));
                const processing = parseFloat((row["Processing (bbl)"] || "0").replace(/,/g, ''));
                const readyTanks = parseInt(row["Ready Tanks"] || "0");
                
                const tankCapacity = parseFloat(params.tankCapacity || 600000);
                const numTanks = parseInt(params.numTanks || 12);
                const tankUtilization = (openingStock / (tankCapacity * numTanks)) * 100;
                
                let cargoTypes = [];
                let totalArrivals = 0;
                let certifiedStock = 0;
                
                if (simulationData.simulation_log) {
                    const currentDate = dateStr.split(' ')[0];
                    
                    const dailyStatusLog = simulationData.simulation_log.find(log => {
                        if (!log.Timestamp || !log.Event) return false;
                        const logDate = log.Timestamp.split(' ')[0];
                        return log.Event === 'DAILY_STATUS' && logDate === currentDate;
                    });
                    
                    if (dailyStatusLog && dailyStatusLog.Message) {
                        const totalStockMatch = dailyStatusLog.Message.match(/TOTAL:\s*([\d,]+)\s*bbl/);
                        if (totalStockMatch) {
                            certifiedStock = parseFloat(totalStockMatch[1].replace(/,/g, ''));
                        }
                    }
                    
                    const arrivalsOnThisDay = simulationData.simulation_log.filter(log => {
                        if (!log.Timestamp || !log.Event) return false;
                        const logDate = log.Timestamp.split(' ')[0];
                        return log.Event === 'ARRIVAL' && logDate === currentDate;
                    });
                    
                    if (arrivalsOnThisDay.length > 0 && simulationData.cargo_report) {
                        arrivalsOnThisDay.forEach(arrival => {
                            const cargoMatch = simulationData.cargo_report.find(cargo => 
                                cargo["Arrival Date"] === currentDate && 
                                cargo["Vessel Name"] === arrival.Cargo
                            );
                            if (cargoMatch) {
                                cargoTypes.push(cargoMatch["Cargo Type"]);
                                totalArrivals += parseFloat((cargoMatch["Total Volume Discharged (bbl)"] || "0").replace(/,/g, ''));
                            }
                        });
                    }
                }
                
                return {
                    day: index + 1,
                    date: dateStr,
                    start_inventory: openingStock,
                    end_inventory: closingStock,
                    processing: processing,
                    ready_tanks: readyTanks,
                    tank_utilization: tankUtilization,
                    cargo_type: cargoTypes.length > 0 ? cargoTypes.join(' + ') : null,
                    arrivals: totalArrivals,
                    certified_stock: certifiedStock,
                    ...Object.keys(row)
                        .filter(key => key.startsWith('Tank') && key.length <= 6)
                        .reduce((acc, key) => ({ ...acc, [key]: row[key] }), {})
                };
            });
        }

        const optimizationSummary = currentResults ? currentResults.optimization_results : null;
        currentResults = simulationData;
        if (optimizationSummary) {
            currentResults.optimization_results = optimizationSummary;
        }

        // Display results
        displayResults(currentResults);
        displayInventoryTracking(currentResults.simulation_data); 

        Utils.showResults();
        showTab('simulation', document.querySelector('.tab'));
        
        // --- 4. Handle Delayed CSV File Downloads (The Fix) ---
        if (simulationData.csv_files && Object.keys(simulationData.csv_files).length > 0) {
            
            console.log('Attempting sequential download of CSV files to avoid browser block.');

            const filesToDownload = Object.entries(simulationData.csv_files);
            
            // Helper function to create a delay
            const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

            // Execute downloads sequentially with a small delay
            (async () => {
                for (const [filename, url] of filesToDownload) {
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = filename;
                    link.style.display = 'none';
                    document.body.appendChild(link);
                    
                    // Trigger the download
                    link.click();
                    document.body.removeChild(link);

                    // Introduce a small delay (50ms)
                    await delay(250); 
                }
                
                // Show success message after all downloads have been initiated
                const fileCount = filesToDownload.length;
                console.log(`✅ Download process initiated for ${fileCount} CSV files. Check your downloads folder.`);
            })();
            
        } else if (simulationData.download_url) {
            // Legacy: Handle old zip file format
            window.location.href = simulationData.download_url;
        }

    } catch (error) {
        console.error('Simulation error:', error);
        alert('Simulation failed: ' + error.message);
    } finally {
        Utils.showLoading(false);
    }
}
/**
 * DISPLAY RESULTS
 */
function displayResults(data) {

    const now = new Date();
    const timestampElement = document.getElementById('reportTimestamp');
    if (timestampElement) {
        timestampElement.textContent = `Report generated on: ${now.toLocaleString()}`;
    }
    // END OF NEW CODE
    
   
    // Display metrics
    const metricsContainer = document.getElementById('metricsContainer');
    metricsContainer.innerHTML = '<h3> Performance Metrics</h3>';

    if (data.metrics) {
        const metricsDiv = document.createElement('div');
        metricsDiv.className = 'metrics-grid';
        // Safety checks added for metrics before calling .toFixed()
        const processingEfficiency = data.metrics.processing_efficiency ? data.metrics.processing_efficiency.toFixed(1) : 'N/A';
        const avgUtilization = data.metrics.avg_utilization ? data.metrics.avg_utilization.toFixed(1) : 'N/A';

        metricsDiv.innerHTML = `
            <div class="metric-card">
                <h4>Processing Efficiency</h4>
                <p class="metric-value">${processingEfficiency}%</p>
            </div>
            <div class="metric-card">
                <h4>Total Processed</h4>
                <p class="metric-value">${data.metrics.total_processed ? data.metrics.total_processed.toLocaleString() : 'N/A'} bbl</p>
            </div>
            <div class="metric-card">
                <h4>Critical Days</h4>
                <p class="metric-value">${data.metrics.critical_days} days</p>
            </div>
            <div class="metric-card">
                <h4>Tank Utilization</h4>
                <p class="metric-value">${avgUtilization}%</p>
            </div>
            <div class="metric-card">
                <h4>Clash Days</h4>
                <p class="metric-value">${data.metrics.clash_days} days</p>
            </div>
            <div class="metric-card">
                <h4>Sustainable</h4>
                <p class="metric-value">${data.metrics.sustainable_processing ? '✅ Yes' : '❌ No'}</p>
            </div>
        `;
        metricsContainer.appendChild(metricsDiv);
    }

    // Display cargo report
    displayCargoReport(data);

    // Display Simulation Data:
    displaySimulationLog(data);


    // Display daily report
    displayDailyReport(data);
}

/**
 * Helper function to safely parse DD/MM/YYYY HH:MM into a Date object.
 */
function safeParseDate(dateStr) {
    if (!dateStr) return null;
    try {
        // Expected format: DD/MM/YYYY HH:MM
        const parts = dateStr.match(/(\d{2})\/(\d{2})\/(\d{4})\s+(\d{2}):(\d{2})/);
        if (parts) {
            // Reorder to YYYY, MM, DD, HH, MM (MM is 1-indexed, but Date constructor uses 0-indexed month)
            // parts[1]=DD, parts[2]=MM, parts[3]=YYYY, parts[4]=HH, parts[5]=MM
            return new Date(parts[3], parts[2] - 1, parts[1], parts[4], parts[5]);
        }
    } catch (e) {
        console.error("Error parsing date:", dateStr, e);
    }
    return null;
}

/**
 * * DISPLAY DAILY REPORT - Updated with Time Range Fix
 */
function displayDailyReport(results) {
    const container = document.getElementById('dailyReportContainer');

    if (!results.simulation_data || results.simulation_data.length === 0) {
        container.innerHTML = '<p>No daily report data available</p>';
        return;
    }

    // Get tank capacity and calculate usable capacity
    const tankCapacity = parseFloat(results.parameters?.tankCapacity || 600000);
    const deadBottom = parseFloat(results.parameters?.deadBottom1 || 10000);
    const bufferVolume = parseFloat(results.parameters?.bufferVolume || 500);
    const usableCapacity = tankCapacity - deadBottom - (bufferVolume / 2);

    let tableHTML = `
        <h3>📊 Daily Operations Report</h3>
        <table class="schedule-table">
            <thead>
                <tr>
                    <th>Day</th>
                    <th>Date / Time Range</th>
                    <th>Open Inventory</th>
                    <th style="background-color: #d4edda;">Cert Stk</th>
                    <th style="background-color: #fff3cd;">Uncert Stk</th>
                    <th>Processing</th>
                    <th>Closing Inventory</th>
                    <th>Tank Util %</th>
                    <th>Cargo Arrival</th>
                </tr>
            </thead>
            <tbody>
    `;

    results.simulation_data.forEach((dayData, dayIndex) => {
        
        // --- START CRITICAL JAVASCRIPT FIX ---
        
        let dateToDisplay = dayData.date; // Default to the stored start date/time (e.g., 11/8/2025 08:00)
        
        // Use the safe parser for the day start time
        const dayStartsAt = safeParseDate(dayData.date); 
        
        // Check if this is the last day and if it was a partial day
        if (dayIndex === results.simulation_data.length - 1 && dayStartsAt) {
            
            // Find the precise timestamp from the DAILY_END log for the last entry
            const lastLogEntry = results.simulation_log
                .filter(log => log.Event === 'DAILY_END')
                // Sort by timestamp (descending) and take the first one
                .sort((a, b) => safeParseDate(b.Timestamp) - safeParseDate(a.Timestamp))[0]; 
            
            if (lastLogEntry) {
                // Use the safe parser for the end time
                const endDateTime = safeParseDate(lastLogEntry.Timestamp);
                
                if (endDateTime) {
                    // Format start and end date/time (using 2-digit format for everything)
                    const startDayMonth = dayStartsAt.toLocaleDateString('en-GB', {day: '2-digit', month: '2-digit'});
                    const startTime = dayStartsAt.toLocaleTimeString('en-GB', {hour: '2-digit', minute: '2-digit', hour12: false});
                    
                    const endDayMonth = endDateTime.toLocaleDateString('en-GB', {day: '2-digit', month: '2-digit'});
                    const endTime = endDateTime.toLocaleTimeString('en-GB', {hour: '2-digit', minute: '2-digit', hour12: false});
                    
                    // Reformat the date column as the desired range
                    dateToDisplay = `${startDayMonth} ${startTime} to ${endDayMonth} ${endTime}`;
                }
            }
        }
        
        // --- END CRITICAL JAVASCRIPT FIX ---

        // Use certified stock from DAILY_STATUS log if available
        let certStock = dayData.certified_stock || 0;
        
        // Calculate uncertified stock
        const uncertStock = Math.max(0, dayData.start_inventory - certStock);
        
        const cargoInfo = dayData.cargo_type ? `${dayData.cargo_type} (${Utils.formatNumber(dayData.arrivals)})` : '-';
        const tankUtilization = dayData.tank_utilization ? dayData.tank_utilization.toFixed(1) + '%' : 'N/A';
        
        let processingToShow = dayData.processing;
        if (dayData.expected_processing_resumed && dayData.expected_processing_resumed > 0) {
            processingToShow = dayData.expected_processing_resumed;
        }

        tableHTML += `
            <tr>
                <td><strong>${dayData.day}</strong></td>
                <td>${dateToDisplay}</td>
                <td style="color: #007bff;">${Utils.formatNumber(dayData.start_inventory)}</td>
                <td style="color: #28a745; font-weight: bold;">${Utils.formatNumber(certStock)}</td>
                <td style="color: #856404; font-weight: bold;">${Utils.formatNumber(uncertStock)}</td>
                <td style="color: #dc3545;">${Utils.formatNumber(processingToShow)}</td>
                <td style="color: #28a745;">${Utils.formatNumber(dayData.end_inventory)}</td>
                <td style="color: #6f42c1;">${tankUtilization}</td>
                <td>${cargoInfo}</td>
            </tr>
        `;
    });

    tableHTML += '</tbody></table>';
    container.innerHTML = tableHTML;
}


/**
 * DISPLAY SIMULATION LOG
 */
function displaySimulationLog(results) {
    const container = document.getElementById('simulationLogContainer');
    
    if (!container) {
        const logContainer = document.createElement('div');
        logContainer.id = 'simulationLogContainer';
        const dailyReport = document.getElementById('dailyReportContainer');
        if (dailyReport && dailyReport.parentNode) {
            dailyReport.parentNode.insertBefore(logContainer, dailyReport);
        }
        return displaySimulationLog(results);
    }

    if (!results.simulation_log || results.simulation_log.length === 0) {
        container.innerHTML = '<p>No simulation log available</p>';
        return;
    }

    // Get tank column names
    const tankColumns = Object.keys(results.simulation_log[0]).filter(key => /^Tank\d+$/.test(key));

    // --- START FIX: Add table-layout: fixed and specific column widths to TH elements ---
    let tableHTML = `
        <h3> Detailed Simulation Log</h3>
        <table class="schedule-table" style="font-size: 0.85em; table-layout: fixed; width: 100%;">
            <thead>
                <tr>
                    <th style="width: 13%;">Timestamp</th>
                    <th style="width: 8%;">Level</th>
                    <th style="width: 11%;">Event</th>
                    <th style="width: 7%;">Tank</th>
                    <th style="width: 10%;">Cargo</th>
                    <th style="text-align: left; width: 51%;">Message</th>
                </tr>
            </thead>
            <tbody>
    `;
    // --- END FIX ---

    results.simulation_log.forEach((logEntry, index) => {
        let levelColor = '#000';
        if (logEntry.Level === 'Success') levelColor = '#28a745';
        else if (logEntry.Level === 'Warning') levelColor = '#ffc107';
        else if (logEntry.Level === 'Danger') levelColor = '#dc3545';
        else if (logEntry.Level === 'Info') levelColor = '#007bff';

        let message = logEntry.Message;
        
        // Add READY count for READY events
        if (logEntry.Event && logEntry.Event.startsWith('READY')) {
            let readyCount = 0;
            
            // Look at next row if available
            if (index + 1 < results.simulation_log.length) {
                const nextRow = results.simulation_log[index + 1];
                readyCount = tankColumns.filter(col => nextRow[col] === 'READY').length;
            } else {
                readyCount = tankColumns.filter(col => logEntry[col] === 'READY').length;
            }
            
            if (readyCount > 0) {
                // Keep the logic to append the message, this is correct
                message = `${message} No of READY tanks : ${readyCount}`;
            }
        }

        tableHTML += `
            <tr>
                <td style="word-wrap: break-word;">${logEntry.Timestamp}</td>
                <td style="color: ${levelColor}; font-weight: bold; word-wrap: break-word;">${logEntry.Level}</td>
                <td style="word-wrap: break-word;">${logEntry.Event}</td>
                <td style="word-wrap: break-word;">${logEntry.Tank || '-'}</td>
                <td style="word-wrap: break-word;">${logEntry.Cargo || '-'}</td>
                
                <td style="text-align: left; word-wrap: break-word;">${message}</td>
                </tr>
        `;
    });

    tableHTML += '</tbody></table></div>';
    container.innerHTML = tableHTML;
}


/**
 * BUFFER ANALYSIS
 */
async function calculateBuffer() {
    try {
        Utils.showLoading(true);

        const params = collectFormData();

        const response = await fetch(API_ENDPOINTS.BUFFER_ANALYSIS, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });

        if (!response.ok) {
            throw new Error('Buffer analysis request failed');
        }

        const bufferResults = await response.json();

        displayBufferAnalysis(bufferResults);
        Utils.showResults();
        showTab('buffer', document.querySelectorAll('.tab')[1]);

    } catch (error) {
        console.error('Buffer analysis error:', error);
        alert('Buffer analysis failed: ' + error.message);
    } finally {
        Utils.showLoading(false);
    }
}

/**
 * DISPLAY BUFFER ANALYSIS
 */
function displayBufferAnalysis(bufferResults) {
    const container = document.getElementById('bufferResults');

    let html = '<h3>🛡️ Buffer Analysis Report</h3>';

    if (bufferResults && Object.keys(bufferResults).length > 0) {
        html += '<div class="buffer-scenarios">';

        Object.entries(bufferResults).forEach(([scenarioKey, scenario]) => {
            const adequateText = scenario.adequate_current ? '✅ Adequate' : '❌ Insufficient';
            const adequateColor = scenario.adequate_current ? '#28a745' : '#dc3545';

            html += `
                <div class="scenario-card" style="border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px;">
                    <h4>${scenario.description}</h4>
                    <div class="scenario-details">
                        <p><strong>Lead Time:</strong> ${scenario.lead_time.toFixed(1)} days</p>
                        <p><strong>Buffer Needed:</strong> ${Utils.formatNumber(scenario.buffer_needed)} barrels</p>
                        <p><strong>Tanks Required:</strong> ${scenario.tanks_needed} tanks</p>
                        <p><strong>Current Capacity:</strong> <span style="color: ${adequateColor}; font-weight: bold;">${adequateText}</span></p>
                        ${scenario.additional_tanks > 0 ?
                            `<p style="color: #dc3545;"><strong>Additional Tanks Needed:</strong> ${scenario.additional_tanks}</p>` :
                            '<p style="color: #28a745;"><strong>No additional tanks needed</strong></p>'
                        }
                    </div>
                </div>
            `;
        });

        html += '</div>';
    } else {
        html += '<p>No buffer analysis data available</p>';
    }

    container.innerHTML = html;
}

/**
 * CARGO OPTIMIZATION
 */
async function optimizeTanks() {
    try {
        Utils.showLoading(true);

        const params = collectFormData();

        const response = await fetch(API_ENDPOINTS.CARGO_OPTIMIZATION, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });

        if (!response.ok) {
            throw new Error('Optimization request failed');
        }

        const optimizationResults = await response.json();

        displayCargoOptimizationResults(optimizationResults);
        Utils.showResults();
        showTab('optimization', document.querySelectorAll('.tab')[2]);

    } catch (error) {
        console.error('Cargo optimization error:', error);
        alert('Optimization failed: ' + error.message);
    } finally {
        Utils.showLoading(false);
    }
}

/**
 

/**
 * SHOW TANK STATUS
 */
async function showTankStatus() {
    if (!currentResults) {
        alert('Please run a simulation first');
        return;
    }

    try {
        Utils.showLoading(true);

        const response = await fetch(API_ENDPOINTS.EXPORT_TANK_STATUS, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(currentResults)
        });

        if (!response.ok) {
            throw new Error('Tank status export failed');
        }

        const result = await response.json();
        alert(`✅ Tank status exported: ${result.filename}`);

    } catch (error) {
        console.error('Tank status error:', error);
        alert('Tank status export failed: ' + error.message);
    } finally {
        Utils.showLoading(false);
    }
}

/**
 * EXPORT SIMULATION REPORT
 */
async function exportSimulationReport() {
    try {
        Utils.showLoading(true);

        if (!currentResults) {
            alert('Please run a simulation first before exporting.');
            Utils.showLoading(false);
            return;
        }

        const response = await fetch(API_ENDPOINTS.EXPORT_TANK_STATUS, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(currentResults)
        });

        if (!response.ok) {
            throw new Error('Export failed');
        }

        const result = await response.json();
        alert(`✅ Simulation report exported: ${result.filename}`);

    } catch (error) {
        console.error('Export error:', error);
        alert('Export failed: ' + error.message);
    } finally {
        Utils.showLoading(false);
    }
}

/**
 * DISPLAY CARGO REPORT
 */
function displayCargoReport(data) {
    const container = document.getElementById('cargoReportContainer');
    if (!container) return;

    if (!data.cargo_report || data.cargo_report.length === 0) {
        container.innerHTML = '<h3>🚢 Cargo Schedule Report</h3><p><em>No cargo schedule available</em></p>';
        return;
    }

    const cargoReport = data.cargo_report;

    let html = '<h3>🚢 Cargo Schedule Report</h3>';

    html += '<p><em>Detailed cargo timeline with load port, departure, arrival, and discharge times</em></p>';
    html += '<div class="cargo-schedule-table">';
    html += '<table class="data-table">';

    html += '<thead><tr>';
    html += '<th>BERTH</th>';
    html += '<th>CARGO NAME</th>';
    html += '<th>CARGO TYPE</th>';
    html += '<th>SIZE</th>';
    html += '<th>L.PORT TIME</th>';
    html += '<th>ARRIVAL</th>';
    html += '<th>PUMPING</th>';
    html += '<th>DEP.TIME</th>';
    html += '</tr></thead><tbody>';

    cargoReport.forEach(cargo => {
        html += '<tr>';
        html += `<td>${cargo.berth || 'N/A'}</td>`;
        html += `<td>${cargo.vessel_name || 'N/A'}</td>`;
        html += `<td>${cargo.type || 'N/A'}</td>`;
        html += `<td>${Utils.formatNumber(cargo.size) || '0'}</td>`;
        html += `<td>${cargo.load_port_time || 'N/A'}</td>`;
        html += `<td>${cargo.arrival_time || 'N/A'}</td>`;
        html += `<td>${cargo.pumping_days ? cargo.pumping_days.toFixed(1) : 'N/A'}</td>`;
        html += `<td>${cargo.dep_unload_port || 'N/A'}</td>`;
        html += '</tr>';
    });

    html += '</tbody></table></div>';

    container.innerHTML = html;
}

/**
 * SHOW TAB
 */
function showTab(tabId, tabButton) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });

    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });

    // Show selected tab content and activate button
    document.getElementById(tabId).classList.add('active');
    if (tabButton) tabButton.classList.add('active');
}

/**
 * Validate inventory range inputs in real-time - Updated for dynamic tanks
 */
function validateInventoryRange() {
    const minInventory = parseFloat(document.getElementById('minInventory').value) || 0;
    const maxInventory = parseFloat(document.getElementById('maxInventory').value) || 0;
    const messageDiv = document.getElementById('inventoryValidationMessage');
    // Count actual tank boxes in DOM instead of input field
    const actualTankCount = document.querySelectorAll('.tank-box').length;

    let isValid = true;
    let message = '';
    let messageType = 'success';

    if (minInventory >= maxInventory) {
        isValid = false;
        message = '❌ Minimum inventory must be less than maximum inventory';
        messageType = 'error';
    } else if (minInventory < 0 || maxInventory < 0) {
        isValid = false;
        message = '❌ Inventory values cannot be negative';
        messageType = 'error';
    } else {
        // Calculate current inventory for all actual tanks
        let currentInventory = 0;
        const tankLevelInputs = document.querySelectorAll('input[id*="Level"]');
        tankLevelInputs.forEach(input => {
            if (input.id.includes('tank') && input.id.includes('Level')) {
                const tankNumber = input.id.replace('tank', '').replace('Level', '');
                const tankLevel = parseFloat(input.value) || 0;
                const deadBottom = parseFloat(document.getElementById(`deadBottom${tankNumber}`)?.value) || 10000;
                currentInventory += Math.max(0, tankLevel - deadBottom);
            }
        });

        if (currentInventory < minInventory) {
            isValid = false;
            message = `⚠️ Current inventory (${currentInventory.toLocaleString()} bbl) is below minimum (${minInventory.toLocaleString()} bbl)`;
            messageType = 'warning';
        } else if (currentInventory > maxInventory) {
            isValid = false;
            message = `⚠️ Current inventory (${currentInventory.toLocaleString()} bbl) is above maximum (${maxInventory.toLocaleString()} bbl)`;
            messageType = 'warning';
        } else {
            message = `✅ Current inventory: ${currentInventory.toLocaleString()} bbl (Range: ${minInventory.toLocaleString()} - ${maxInventory.toLocaleString()} bbl) - ${actualTankCount} tanks`;
            messageType = 'success';
        }
    }

    // Display message
    if (messageDiv) {
        messageDiv.style.display = 'block';
        messageDiv.innerHTML = message;

        if (messageType === 'error') {
            messageDiv.style.backgroundColor = '#f8d7da';
            messageDiv.style.color = '#721c24';
            messageDiv.style.border = '1px solid #f5c6cb';
        } else if (messageType === 'warning') {
            messageDiv.style.backgroundColor = '#fff3cd';
            messageDiv.style.color = '#856404';
            messageDiv.style.border = '1px solid #ffeaa7';
        } else {
            messageDiv.style.backgroundColor = '#d1edff';
            messageDiv.style.color = '#0c5460';
            messageDiv.style.border = '1px solid #bee5eb';
        }
    }

    return isValid;
}

/**
 * INVENTORY button click handler
 */
function checkInventoryRange() {
    Utils.showLoading(true);

    const params = collectFormData();

    fetch('/api/validate_inventory_range', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        })
        .then(response => response.json())
        .then(data => {
            Utils.showLoading(false);

            if (data.success) {
                alert(`✅ INVENTORY RANGE VALIDATION PASSED\n\n${data.message}\n\nYou can proceed with simulation.`);
            } else {
                alert(`❌ INVENTORY RANGE VALIDATION FAILED\n\n${data.message}\n\nPlease adjust your inventory range or tank levels.`);
            }
        })
        .catch(error => {
            Utils.showLoading(false);
            console.error('Inventory validation error:', error);
            alert('❌ Error validating inventory range. Please try again.');
        });
}

/**
 * Display inventory tracking results
 */
function displayInventoryTracking(inventoryData) {
    const container = document.getElementById('inventoryResults');
    if (!container || !inventoryData || inventoryData.length === 0) {
        if (container) container.innerHTML = '<p>No inventory tracking data available.</p>';
        return;
    }
    
    // Setup for Chart.js
    const ctx = document.getElementById('inventoryChart').getContext('2d');
    const labels = inventoryData.map(d => `Day ${d.day}`);
    const dataPoints = inventoryData.map(d => d.end_inventory);

    // Destroy existing chart if it exists to prevent conflicts
    if (window.myInventoryChart) {
        window.myInventoryChart.destroy();
    }

    // Create the chart
    window.myInventoryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'End of Day Inventory (bbl)',
                data: dataPoints,
                borderColor: '#007bff',
                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value, index, values) {
                            return value.toLocaleString() + ' bbl';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Enhanced runSimulation function with inventory validation
 */
function runSimulationWithInventoryCheck() {
    // First validate inventory range
    const minInventory = parseFloat(document.getElementById('minInventory').value) || 0;
    const maxInventory = parseFloat(document.getElementById('maxInventory').value) || 0;

    if (minInventory > 0 || maxInventory > 0) {
        if (minInventory >= maxInventory) {
            alert('❌ SIMULATION BLOCKED\n\nMinimum inventory must be less than maximum inventory.\nPlease fix inventory range before running simulation.');
            return;
        }
    }

    // Proceed with normal simulation
    runSimulation();
}

/**
 * Update collectSimulationParams to include inventory range
 */
function collectSimulationParamsWithInventory() {
    const params = collectFormData();

    // Add inventory range parameters
    params.minInventory = parseFloat(document.getElementById('minInventory').value) || 0;
    params.maxInventory = parseFloat(document.getElementById('maxInventory').value) || 0;

    return params;
}

// CORRECT - handles file download


/**
 * Enhanced export function to handle inventory data
 */
function exportSimulationReportWithInventory() {
    if (!currentResults) {
        alert('Please run a simulation first before exporting.');
        return;
    }

    Utils.showLoading(true);

    fetch('/api/export_tank_status', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(currentResults)
        })
        .then(response => response.json())
        .then(data => {
            Utils.showLoading(false);
            if (data.success) {
                alert(`✅ COMPLETE EXPORT SUCCESSFUL\n\nFile: ${data.filename}\n\nFeatures included:\n${data.features?.join('\n') || 'All fixed requirements implemented'}\n\nIncluding INVENTORY sheet with real-time graph!`);
            } else {
                alert(`❌ Export failed: ${data.error}`);
            }
        })
        .catch(error => {
            Utils.showLoading(false);
            console.error('Export error:', error);
            alert('❌ Export failed. Please try again.');
        });
}

function initializeAutoSave() {
    // Get all input and select elements
    const inputs = document.querySelectorAll('input, select');
    
    inputs.forEach(input => {
        // Add event listeners for different input types
        if (input.type === 'number' || input.type === 'text') {
            // For text/number inputs, save on blur (when user finishes editing)
            input.addEventListener('blur', autoSaveInputs);
            // Also save on input change with debouncing
            let timeout;
            input.addEventListener('input', () => {
                clearTimeout(timeout);
                timeout = setTimeout(autoSaveInputs, 1000); // Save after 1 second of no typing
            });
        } else {
            // For select, radio, checkbox - save immediately on change
            input.addEventListener('change', autoSaveInputs);
        }
    });
    
    console.log(`Auto-save initialized for ${inputs.length} inputs`);
}

// ADD SAVE STATUS INDICATOR (Optional visual feedback)
function showSaveStatus(status) {
    // Create or update save status indicator
    let indicator = document.getElementById('saveIndicator');
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.id = 'saveIndicator';
        indicator.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 8px 12px;
            background: #28a745;
            color: white;
            border-radius: 4px;
            font-size: 12px;
            z-index: 1000;
            transition: opacity 0.3s;
        `;
        document.body.appendChild(indicator);
    }
    
    if (status === 'saved') {
        indicator.textContent = '✓ Saved';
        indicator.style.opacity = '1';
        setTimeout(() => {
            indicator.style.opacity = '0';
        }, 2000);
    }
}




// HELPER FUNCTION TO APPLY VALUES
function applyInputValues(inputValues) {

    if (inputValues.numTanks) {
        document.getElementById('numTanks').value = inputValues.numTanks;
        updateTankCount();
     }

    Object.entries(inputValues).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
            if (element.type === 'checkbox') {
                element.checked = value;
            } else {
                element.value = value;
            }
        }
    });

    // FIXED: Recreate crude mix table if saved data exists
    if (inputValues.crudeMixData) {
        recreateCrudeMixTable(inputValues.crudeMixData);
    }


    // Update calculations after loading
    toggleDepartureMode();
    
    validateInventoryRange();
}

async function exportCharts() {
    if (!currentResults) {
        alert('⚠️ Please run a simulation first to generate charts data.');
        return;
    }

    try {
        Utils.showLoading(true);
        document.getElementById('loading').querySelector('p').textContent = 'Generating charts...';
        
        const response = await fetch('/api/export_charts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(currentResults)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const contentType = response.headers.get('content-type');
        
        if (contentType && contentType.includes('application/json')) {
            const result = await response.json();
            if (result.success) {
                alert(`✅ ${result.message}`);
            } else {
                alert(`❌ Charts export failed: ${result.error}`);
            }
        } else {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            
            const disposition = response.headers.get('Content-Disposition');
            let filename = 'charts_export.xlsx';
            if (disposition) {
                const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition);
                if (matches != null && matches[1]) {
                    filename = matches[1].replace(/['"]/g, '');
                }
            }
            
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            alert(`✅ Charts exported: ${filename}`);
        }
        
    } catch (error) {
        console.error('Charts export error:', error);
        alert(`❌ Export failed: ${error.message}`);
    } finally {
        Utils.showLoading(false);
        document.getElementById('loading').querySelector('p').textContent = 'Running simulation...';
    }
}



// CONSOLIDATED INITIALIZATION - Merged from HTML and existing main.js

document.addEventListener('DOMContentLoaded', () => {
    // Load saved inputs first
    autoLoadInputs();
    
    // Initialize calculations (from HTML)
    

    validateInventoryRange();
    //updateRateStepDisplay(); #step fucntion processing rate
    
    // Update tank count to create missing tanks
    setTimeout(() => {
        updateTankCount();
        initializeAutoSave();
    }, 500); // Small delay to ensure all elements are loaded
});


/**
 * FIXED: Show Tank Status - Handle file download properly
 */
async function showTankStatus() {
    if (!currentResults) {
        alert('Please run a simulation first');
        return;
    }

    try {
        Utils.showLoading(true);

        const response = await fetch(API_ENDPOINTS.EXPORT_TANK_STATUS, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(currentResults)
        });

        if (!response.ok) {
            throw new Error('Tank status export failed');
        }

        // Check if response is a file or JSON
        const contentType = response.headers.get('content-type');
        
        if (contentType && contentType.includes('application/json')) {
            // It's JSON - parse it
            const result = await response.json();
            alert(`✅ Tank status exported: ${result.filename}`);
        } else {
            // It's a file - trigger download
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            
            // Get filename from Content-Disposition header or use default
            const disposition = response.headers.get('Content-Disposition');
            let filename = 'tank_status_export.xlsx';
            if (disposition) {
                const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition);
                if (matches != null && matches[1]) {
                    filename = matches[1].replace(/['"]/g, '');
                }
            }
            
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            alert(`✅ Tank status downloaded: ${filename}`);
        }

    } catch (error) {
        console.error('Tank status error:', error);
        alert('Tank status export failed: ' + error.message);
    } finally {
        Utils.showLoading(false);
    }
}

/**
 * FIXED: Export Simulation Report - Handle file download properly
 */
async function exportSimulationReport() {
    try {
        Utils.showLoading(true);

        if (!currentResults) {
            alert('Please run a simulation first before exporting.');
            Utils.showLoading(false);
            return;
        }

        const response = await fetch(API_ENDPOINTS.EXPORT_TANK_STATUS, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(currentResults)
        });

        if (!response.ok) {
            throw new Error('Export failed');
        }

        // Check if response is a file or JSON
        const contentType = response.headers.get('content-type');
        
        if (contentType && contentType.includes('application/json')) {
            // It's JSON - parse it
            const result = await response.json();
            alert(`✅ Simulation report exported: ${result.filename}`);
        } else {
            // It's a file - trigger download
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            
            // Get filename from Content-Disposition header or use default
            const disposition = response.headers.get('Content-Disposition');
            let filename = 'simulation_report.xlsx';
            if (disposition) {
                const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition);
                if (matches != null && matches[1]) {
                    filename = matches[1].replace(/['"]/g, '');
                }
            }
            
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            alert(`✅ Simulation report downloaded: ${filename}`);
        }

    } catch (error) {
        console.error('Export error:', error);
        alert('Export failed: ' + error.message);
    } finally {
        Utils.showLoading(false);
    }
}

/**
 * FIXED: Export Simulation Report With Inventory - Handle file download properly
 */
function exportSimulationReportWithInventory() {
    if (!currentResults) {
        alert('Please run a simulation first before exporting.');
        return;
    }

    Utils.showLoading(true);

    fetch('/api/export_tank_status', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(currentResults)
        })
        .then(async response => {
            if (!response.ok) {
                throw new Error('Export failed');
            }

            // Check if response is a file or JSON
            const contentType = response.headers.get('content-type');
            
            if (contentType && contentType.includes('application/json')) {
                // It's JSON - parse it
                const data = await response.json();
                Utils.showLoading(false);
                if (data.success) {
                    alert(`✅ COMPLETE EXPORT SUCCESSFUL\n\nFile: ${data.filename}\n\nFeatures included:\n${data.features?.join('\n') || 'All fixed requirements implemented'}\n\nIncluding INVENTORY sheet with real-time graph!`);
                } else {
                    alert(`❌ Export failed: ${data.error}`);
                }
            } else {
                // It's a file - trigger download
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                
                // Get filename from Content-Disposition header or use default
                const disposition = response.headers.get('Content-Disposition');
                let filename = 'complete_simulation_report.xlsx';
                if (disposition) {
                    const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition);
                    if (matches != null && matches[1]) {
                        filename = matches[1].replace(/['"]/g, '');
                    }
                }
                
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                Utils.showLoading(false);
                alert(`✅ COMPLETE EXPORT SUCCESSFUL\n\nFile downloaded: ${filename}\n\nIncluding INVENTORY sheet with real-time graph!`);
            }
        })
        .catch(error => {
            Utils.showLoading(false);
            console.error('Export error:', error);
            alert('❌ Export failed. Please try again.');
        });
}

/**
 * FIXED: Export Charts - Handle file download properly
 */
async function exportCharts() {
    // Check if simulation has been run
    if (!currentResults) {
        alert('⚠️ Please run a simulation first to generate charts data.');
        return;
    }

    try {
        // Show loading spinner
        Utils.showLoading(true);
        document.getElementById('loading').querySelector('p').textContent = 'Generating charts...';
        
        // Send simulation results to backend for chart generation
        const response = await fetch('/api/export_charts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(currentResults)
        });

        if (!response.ok) {
            throw new Error('Charts export failed');
        }

        // Check if response is a file or JSON
        const contentType = response.headers.get('content-type');
        
        if (contentType && contentType.includes('application/json')) {
            // It's JSON - parse it
            const result = await response.json();
            if (result.success) {
                alert(`✅ ${result.message}`);
            } else {
                alert(`❌ Charts export failed: ${result.error}`);
            }
        } else {
            // It's a file - trigger download
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            
            // Get filename from Content-Disposition header or use default
            const disposition = response.headers.get('Content-Disposition');
            let filename = 'charts_export.xlsx';
            if (disposition) {
                const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition);
                if (matches != null && matches[1]) {
                    filename = matches[1].replace(/['"]/g, '');
                }
            }
            
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            alert(`✅ Charts exported and downloaded: ${filename}`);
        }
        
    } catch (error) {
        console.error('Charts export error:', error);
        alert(`❌ Charts export error: ${error.message}`);
    } finally {
        // Hide loading spinner
        Utils.showLoading(false);
        document.getElementById('loading').querySelector('p').textContent = 'Running simulation...';
    }
}
/**
 * OPTIMIZE CRUDE MIX SCHEDULE
 */
async function optimizeCrudeMix() {
    try {
        Utils.showLoading(true);
        document.getElementById('loading').querySelector('p').textContent = 'Optimizing crude mix...';

        const params = collectFormData();

        // Extract crude mix data from the table
        const crudeNames = [];
        const crudePercentages = [];
        
        document.querySelectorAll('.crude-mix-row').forEach(row => {
            const nameInput = row.querySelector('.crude-name-input');
            const percentageInput = row.querySelector('.crude-percentage-input');
            
            if (nameInput && percentageInput) {
                const name = nameInput.value.trim();
                const percentage = parseFloat(percentageInput.value) || 0;
                
                if (name && percentage > 0) {
                    crudeNames.push(name);
                    crudePercentages.push(percentage);
                }
            }
        });

        // Add crude mix arrays to params
        params.crude_names = crudeNames;
        params.crude_percentages = crudePercentages;

        console.log('Sending optimization request with:', {
            crudeNames,
            crudePercentages,
            paramsKeys: Object.keys(params)
        });

        const response = await fetch('/api/optimize_crude_mix', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });

        const results = await response.json();

        if (!response.ok || !results.success) {
            const errorMessage = results.error || 'Optimization request failed';
            const errorDetails = results.details ? `\n\nDetails: ${results.details}` : '';
            throw new Error(errorMessage + errorDetails);
        }

        if (!currentResults) {
            currentResults = {};
        }
        currentResults.parameters = params;
        currentResults.parameters.use_optimized_schedule = true;
        currentResults.parameters.optimized_cargo_schedule = results.optimization_results.cargo_schedule;
        currentResults.optimization_results = results.optimization_results;
        
        displayOptimizationResults(results);
        Utils.showResults();
        showTab('simulation', document.querySelector('.tab'));
        displayCargoReport(currentResults);
        
        alert('✅ Crude mix optimization successful! Click "Run Simulation" to see it in action.');

    } catch (error) {
        console.error('Crude mix optimization error:', error);
        alert('Optimization Failed:\n' + error.message);
    } finally {
        Utils.showLoading(false);
        document.getElementById('loading').querySelector('p').textContent = 'Running simulation...';
    }
}
/**
 * EXPORT SOLVER REPORT
 * Calls the backend to generate and download a .txt report from the last optimization run.
 */
async function exportSolverReport() {
    // 1. Check if optimization results exist
    if (!currentResults || !currentResults.optimization_results) {
        alert('⚠️ Please run a crude mix optimization first to generate a solver report.');
        return;
    }

    try {
        // 2. Show loading indicator to the user
        Utils.showLoading(true);
        document.getElementById('loading').querySelector('p').textContent = 'Generating solver report...';
        
        // 3. Call the new API endpoint
        const response = await fetch('/api/export_solver_report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(currentResults.optimization_results) 
        });

        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }

        // 4. This block handles the actual file download
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        
        // Get filename from server or use a default
        const disposition = response.headers.get('Content-Disposition');
        let filename = 'solver_report.txt';
        if (disposition) {
            const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition);
            if (matches != null && matches[1]) {
                filename = matches[1].replace(/['"]/g, '');
            }
        }
        
        a.download = filename;
        document.body.appendChild(a);
        a.click(); // This triggers the browser download
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        alert(`✅ Solver report downloaded successfully: ${filename}`);
        
    } catch (error) {
        console.error('Solver report export error:', error);
        alert(`❌ Solver report export error: ${error.message}`);
    } finally {
        // 5. Hide loading indicator
        Utils.showLoading(false);
        document.getElementById('loading').querySelector('p').textContent = 'Running simulation...';
    }
}



/**
 * DISPLAY OPTIMIZATION RESULTS (helper function)
 */


function displayOptimizationResults(results) {
    // We will display the results in the main simulation tab for now
    const metricsContainer = document.getElementById('metricsContainer');
    
    if (results.success && results.optimization_results) {
        const opt = results.optimization_results;
        
        let html = '<h3>🧪 Crude Mix Optimization Summary</h3>';
        html += '<div class="metrics-grid">';
        
        html += Utils.createMetricCard(
            'Total Charter Cost',
            opt.total_charter_cost,
            'Based on vessel rates'
        );
        
        html += Utils.createMetricCard(
            'Cargoes Scheduled',
            opt.total_cargoes,
            'To meet inventory and mix targets'
        );
        
        html += Utils.createMetricCard(
            'Solver Status',
            opt.solver_status,
            'Result from the optimization engine'
        );
        
        html += '</div>';
        metricsContainer.innerHTML = html;
    }
}

// Make all functions globally available
window.populateTankLevels = populateTankLevels;
window.toggleDepartureMode = toggleDepartureMode;
window.applyDefaultDeadBottom = applyDefaultDeadBottom;
window.autoSaveInputs = autoSaveInputs;
window.autoLoadInputs = autoLoadInputs;
window.runSimulation = runSimulation;
window.calculateBuffer = calculateBuffer;
window.optimizeTanks = optimizeTanks;
window.showTankStatus = showTankStatus;
window.exportSimulationReport = exportSimulationReport;
window.showTab = showTab;
window.validateInventoryRange = validateInventoryRange;
window.checkInventoryRange = checkInventoryRange;
window.runSimulationWithInventoryCheck = runSimulationWithInventoryCheck;
window.exportSimulationReportWithInventory = exportSimulationReportWithInventory;

// Add the moved functions to global window object
window.scrollToTop = scrollToTop;
window.scrollToCargoReport = scrollToCargoReport;
window.scrollToBottom = scrollToBottom;
window.scrollToSimulation = scrollToSimulation;
window.updateTankCount = updateTankCount;
window.addOneTank = addOneTank;
window.addNewTankBox = addNewTankBox;
window.initializeAutoSave = initializeAutoSave;
window.showSaveStatus = showSaveStatus;
window.applyInputValues = applyInputValues;
window.getCurrentTankCount = getCurrentTankCount;
window.exportCharts = exportCharts;
window.optimizeCrudeMix = optimizeCrudeMix; 
window.addCrudeRow = addCrudeRow;
window.removeCrudeRow = removeCrudeRow;
window.updateCrudeMix = updateCrudeMix;
window.resetCrudeMix = resetCrudeMix
window.updateTankCapacities = updateTankCapacities;
window.collectCrudeMixData = collectCrudeMixData;
window.recreateCrudeMixTable = recreateCrudeMixTable;
window.exportSolverReport = exportSolverReport;