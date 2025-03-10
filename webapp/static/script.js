let map;
let drawingManager;
let selectedArea;
let geocoder;
let spinner = document.getElementById("spinner"); // Assuming you have a spinner element with ID "spinner"
let gridCellSizeMeters = 20
let gridDimension = 5
let radius = gridCellSizeMeters * gridDimension; // Adjust based on your grid size
// Function to handle the common behavior for both map and selectArea click
function handleMapClick(event) {
    // Clear previous selection area
    if (selectedArea) {
        selectedArea.setMap(null);
    }
    infoWindow.close()

    const clickedLatLng = event.latLng;
    selectedBounds = calculateBounds(clickedLatLng, radius);
    
    // Create a new rectangle to represent the selection area
    selectedArea = new google.maps.Rectangle({
        bounds: selectedBounds.selectedBounds,
        editable: false,
        draggable: false,
        map: map
    });

    google.maps.event.addListener(selectedArea, 'click', handleMapClick);

    map.fitBounds(selectedBounds.wideBounds);
    document.getElementById("detectButton").style.display = 'block';
}

// Initialize the map,
function initMap() { 

    try {
        map = new google.maps.Map(document.getElementById('map'), {
            zoom: 19,
            center: { lat: 34.4289741, lng: -118.5975942 },  // Default to San Francisco
            mapTypeId: 'satellite',
            tilt: 0, // Set tilt to 0 to prevent orthographic view
            disableDefaultUI: true,
            zoomControl: true
        });
    } catch (error) {
        console.error("Error initializing the map:", error);
        setTimeout(initMap, 1000);
    }

    

    // Use ipinfo.io to get user's location based on their IP
    fetch('https://ipinfo.io?token='+ipinfo_token)  // Replace YOUR_TOKEN with your actual token from ipinfo.io
    .then(response => response.json())
    .then(data => {
        // Parse the location from the response
        const location = data.loc.split(','); // loc is a comma-separated string like "latitude,longitude"
        const latitude = parseFloat(location[0]);
        const longitude = parseFloat(location[1]);

        // Center the map on the user's location
        map.setCenter({ lat: latitude, lng: longitude });
        map.setZoom(19); // Adjust zoom level as needed
    })
    .catch(error => {
        console.error('Error fetching location from ipinfo.io:', error);
    });

    // The info window that will display the probabilities table and feedback buttons
    infoWindow = new google.maps.InfoWindow({
        content: `
            <div id="infoWindowContent">
                <h3>Area Information</h3>
                <div id="probabilitiesTableContainer"></div>
            </div>
        `
    });
    


    // Initialize the geocoder
    geocoder = new google.maps.Geocoder();

    google.maps.event.addListener(map, 'click', handleMapClick);

    const searchBox = document.getElementById('addressSearch');
    searchBox.addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            geocodeAddress();
        }
    });

    const closeButton = document.getElementById('closeButton');
    closeButton.addEventListener('click', function() {
        controlPanel.style.display = 'none';  // Hide the control panel
    });

    google.maps.event.addListener(map, 'center_changed', function() {
        if (selectedArea) {
            // Get the center of the map
            const centerLatLng = map.getCenter();

            selectedBounds = calculateBounds(centerLatLng, radius);

            // Move the selected area to the new center
            selectedArea.setBounds(selectedBounds.selectedBounds);
        }
    });

}

function calculateBounds(center, sizeInMeters) {
    const earthRadius = 6378137;  // Radius of the Earth in meters
    const lat = center.lat();
    const lng = center.lng();

    // Calculate offsets in latitude and longitude for the given size in meters
    const latOffset = (sizeInMeters / earthRadius) * (180 / Math.PI);
    
    // Longitude offset should also consider the Earth's curvature, with adjustments based on latitude
    const lngOffset = (sizeInMeters / earthRadius) * (180 / Math.PI) / Math.cos(lat * Math.PI / 180);

    // Calculate the bounding box
    const south = lat - latOffset;
    const west = lng - lngOffset;
    const north = lat + latOffset;
    const east = lng + lngOffset;

    // Correct the bin size for 25 meters in latitude and longitude.
    // Latitude: each degree is roughly 111,000 meters at the equator.
    const binSizeLat = (2 * sizeInMeters / gridDimension / earthRadius) * (180 / Math.PI); // 25 meters in latitude
    const binnedLat = Math.round((lat + binSizeLat/2) / binSizeLat) * binSizeLat;

    // Longitude: The distance between longitude lines changes with latitude, so we adjust based on the cosine of the latitude.
    const binSizeLng = Math.round(((2 * sizeInMeters / gridDimension / earthRadius) * (180 / Math.PI) / Math.cos(binnedLat * Math.PI / 180)) * 100000) / 100000;

    // Snap the b  ounds to the nearest bin size
    const binnedNorth = Math.round(north / binSizeLat) * binSizeLat;
    const binnedSouth = Math.round(south / binSizeLat) * binSizeLat;
    const binnedEast = Math.round(east / binSizeLng) * binSizeLng;
    const binnedWest = Math.round(west / binSizeLng) * binSizeLng;
    
    // Return the selected and wide bounds
    return {
        selectedBounds: new google.maps.LatLngBounds(
            new google.maps.LatLng(binnedSouth, binnedWest),
            new google.maps.LatLng(binnedNorth, binnedEast)
        ),
        wideBounds: new google.maps.LatLngBounds(
            new google.maps.LatLng(binnedSouth + 0.15 * latOffset, binnedWest + 0.15 * lngOffset),
            new google.maps.LatLng(binnedNorth - 0.15 * latOffset, binnedEast - 0.15 * lngOffset)
        )
    };
}



function sendToServer(bounds) {
    document.getElementById("detectButton").style.display = 'none';
    const north = bounds.getNorthEast().lat();
    const south = bounds.getSouthWest().lat();
    const east = bounds.getNorthEast().lng();
    const west = bounds.getSouthWest().lng();

    // Calculate the step size for each chunk
    const latStep = (north - south) / 5;
    const lngStep = (east - west) / 5;

    const staticMapUrlBase = 'https://maps.googleapis.com/maps/api/staticmap?' + 
        'zoom=20' +
        '&size=400x400' +
        '&maptype=satellite' +
        '&key=' + maps_api_key;

    // Show the spinner while waiting for the response
    spinner.style.display = 'block'; // Show the spinner

    // Iterate over a 5x5 grid (25 chunks)
    for (let row = 0; row < 5; row++) {
        for (let col = 0; col < 5; col++) {
            const chunkNorth = north - row * latStep;
            const chunkSouth = north - (row + 1) * latStep;
            const chunkEast = west + (col + 1) * lngStep;
            const chunkWest = west + col * lngStep;

            // Calculate the center of the current chunk for the map URL
            const centerLat = (chunkNorth + chunkSouth) / 2;
            const centerLng = (chunkEast + chunkWest) / 2;

            const staticMapUrl = staticMapUrlBase + 
                '&center=' + centerLat + ',' + centerLng 

            fetch(staticMapUrl)
                .then(response => response.blob())
                .then(imageBlob => {
                    const reader = new FileReader();
                    reader.onloadend = function () {
                        const base64Image = reader.result.split(',')[1];
                        const data = {
                            north: chunkNorth,
                            south: chunkSouth,
                            east: chunkEast,
                            west: chunkWest,
                            image: base64Image
                        };

                        fetch('/process-image', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-CSRFToken': csrf_token  // Make sure to send the CSRF token
                            },
                            body: JSON.stringify(data)
                        })
                        .then(response => response.json())
                        .then(responseData => {
                            // Hide the spinner once the response is received
                            spinner.style.display = 'none';
                            displayResponse(responseData);
                        })
                        .catch(error => {
                            // Hide the spinner if there is an error
                            spinner.style.display = 'none';
                            console.error('Error:', error);
                        });
                    };
                    reader.readAsDataURL(imageBlob);
                })
                .catch(error => {
                    // Hide the spinner if there is an error
                    spinner.style.display = 'none';
                    console.error('Error fetching static map:', error);
                });
        }
    }
}


function detectPanels() {
    if (selectedBounds) {
        storedBounds = selectedBounds
        sendToServer(storedBounds.selectedBounds);
    } else {
        console.log("No bounding box selected.");
        alert("Please click on the map to select an area first.");
    }
}

// Create the feedback buttons HTML
function createFeedbackButtons() {
    const feedbackDiv = document.createElement('div');
    feedbackDiv.id= 'feedbackDiv';
    feedbackDiv.innerHTML = `
        <h4>Were there actually panels present?</h4>
        <button id="yesButton" class="feedbackButton" onclick="sendFeedback(1)">👍 Panels</button>
        <button id="noButton" class="feedbackButton" onclick="sendFeedback(0)">👎 No Panels</button>
    `;
    return feedbackDiv;
}

// Create a table dynamically from the probabilities
function createProbabilitiesTable(probabilities) {
    const table = document.createElement("table");
    table.classList.add("probabilities-table");
    table.style.width = "100%";
    table.setAttribute("border", "1");
    
    // Create table header
    const headerRow = document.createElement("tr");
    const headerKey = document.createElement("th");
    headerKey.textContent = "Submodel";
    const headerValue = document.createElement("th");
    headerValue.textContent = "Confidence";
    headerRow.appendChild(headerKey);
    headerRow.appendChild(headerValue);
    table.appendChild(headerRow);

    // Loop through the probabilities and add rows to the table
    for (const key in probabilities) {
        if (probabilities.hasOwnProperty(key)) {
            const row = document.createElement("tr");

            const cellKey = document.createElement("td");
            cellKey.textContent = key;

            const cellValue = document.createElement("td");
            let value = probabilities[key];
            if (typeof value === 'number') {
                value = (value*100).toFixed(0); // Format the number to 3 decimal places
            }
            cellValue.textContent = value+'%';

            row.appendChild(cellKey);
            row.appendChild(cellValue);
            table.appendChild(row);
        }
    }
    return table;
}

function displayResponse(responseData) {
    storedResponseDate = responseData

    // Iterate through responseData and create table rows
    var bounds = new google.maps.LatLngBounds(
        new google.maps.LatLng(responseData.south, responseData.west),
        new google.maps.LatLng(responseData.north, responseData.east)
    )
    var metamodel_probability = responseData.metamodel_probability
    var rectangle = {}
    if (metamodel_probability > 0.5){
        rectangle = new google.maps.Rectangle({
            bounds: bounds,
            editable: false,
            draggable: false,
            map: map,
            strokeColor: '#00FF00', // Change border color to green
            fillColor: '#00FF00'    // Change fill color to green
        });
    } else{
        rectangle = new google.maps.Rectangle({
            bounds: bounds,
            editable: false,
            draggable: false,
            map: map,
            strokeColor: '#FF0000', // Change border color to green
            fillColor: '#FF0000'    // Change fill color to green
        });
    }
    rectangle.metadata = responseData;

    selectedArea.setMap(null);
    if (storedBounds){
        map.fitBounds(storedBounds.wideBounds);
    }
    storedBounds = null
    
    
    // Add click listener to the rectangle
    google.maps.event.addListener(rectangle, 'click', function(event) {
        // Open the infoWindow at the clicked location
        infoWindow.setPosition(event.latLng);
        infoWindow.open(map);
    
        // Generate the probabilities table
        // const probabilitiesTable = createProbabilitiesTable(rectangle.metadata.probabilities);
    
        // Generate the feedback buttons
        const feedbackButtons = createFeedbackButtons();
    
        // Set the entire content of the infoWindow to the table and feedback buttons
        infoWindow.setContent(`
            <div id="infoWindowContent">
                <h2>Detection Confidence = ${(rectangle.metadata.metamodel_probability*100).toFixed(0)}%</h2>
                ${feedbackButtons.outerHTML} <!-- Insert the feedback buttons -->
            </div>
        `);
        // infoWindow.setContent(`
        //     <div id="infoWindowContent">
        //         <h2>Detection Confidence = ${(rectangle.metadata.metamodel_probability*100).toFixed(0)}%</h2>
        //         <h4>Input Model Performance:</h4>
        //         ${probabilitiesTable.outerHTML} <!-- Insert the table -->
        //         ${feedbackButtons.outerHTML} <!-- Insert the feedback buttons -->
        //     </div>
        // `);
    
        // Store the rectangle metadata in a global object to access it later for feedback
        window.currentRectangle = rectangle;
    });

}

function geocodeAddress() {
    const address = document.getElementById("addressSearch").value;

    if (address !== "") {
        geocoder.geocode({ 'address': address }, function(results, status) {
            if (status === 'OK') {
                map.setCenter(results[0].geometry.location);
                map.setZoom(18);
            } else {
                alert('Geocode failed due to: ' + status);
            }
        });
    }
}
function sendFeedback(val) {
    if (window.currentRectangle) {
        // Retrieve the metadata from the clicked rectangle
        document.getElementById("feedbackDiv").style.visibility = 'hidden'
        const rectangleData = window.currentRectangle.metadata;

        // Add the feedback value to the rectangle's metadata
        rectangleData.feedback = val;

        // Send the feedback and rectangle data to the server
        fetch('/store-feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrf_token  // Make sure to send the CSRF token
            },
            body: JSON.stringify(rectangleData)
        })
        .then(response => response.json())
        .then(data => {
            console.log('Feedback stored:', data);
        })
        .catch(err => {
            console.error('Request failed', err);
        });
    }
}


window.onload = function() {
    initMap();
};

// Function to show the control panel popup
function showInfo() {
    // Show the popup when the info button is clicked
    const controlPanel = document.getElementById('controlPanel');
    controlPanel.style.display = 'block';

    // // Add a click event listener to the window to close the popup if clicking outside
    // function closeInfoPanel(event) {
    //     const controlPanel = document.getElementById('controlPanel');
    //     const logo = document.getElementById('logo');
        
    //     // Check if the click is outside the control panel and the info button
    //     if (!controlPanel.contains(event.target) && !logo.contains(event.target)) {
    //         controlPanel.style.display = 'none';
    //         window.removeEventListener('click', closeInfoPanel); // Remove the event listener after closing
    //     }
    // }

    // // Add the click event listener to the window
    // window.addEventListener('click', closeInfoPanel);

}
