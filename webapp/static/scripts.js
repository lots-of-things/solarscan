let map;
let drawingManager;
let selectedArea;
let geocoder;
let spinner = document.getElementById("spinner"); // Assuming you have a spinner element with ID "spinner"

// Initialize the map,
function initMap() { 
    map = new google.maps.Map(document.getElementById('map'), {
        zoom: 19,
        center: { lat: 34.4289741, lng: -118.5975942 },  // Default to San Francisco
        mapTypeId: 'satellite',
        tilt: 0, // Set tilt to 0 to prevent orthographic view
        disableDefaultUI: false,
        zoomControl: true
    });

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


    // Initialize the geocoder
    geocoder = new google.maps.Geocoder();

    google.maps.event.addListener(map, 'click', function(event) {
        // Clear previous selection area
        if (selectedArea) {
            selectedArea.setMap(null);
        }

        const clickedLatLng = event.latLng;
        const radius = 25; 
        bounds = calculateBounds(clickedLatLng, radius);
        
        // Create a new rectangle to represent the selection area
        selectedArea = new google.maps.Rectangle({
            bounds: bounds.selectedBounds,
            editable: false,
            draggable: false,
            map: map
        });

        map.fitBounds(bounds.wideBounds);

        // Clear previous table and hide response content
        const resultContent = document.getElementById("resultContent");
        resultContent.innerHTML = '';
        document.getElementById("responseContent").style.display = 'none';
        document.getElementById("feedbackContainer").style.display = 'none';
    });

    const searchBox = document.getElementById('addressSearch');
    searchBox.addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            geocodeAddress();
        }
    });

    // Event listeners for feedback buttons
    document.getElementById("yesButton").addEventListener("click", function() {
        sendFeedback(1);
    });

    document.getElementById("noButton").addEventListener("click", function() {
        sendFeedback(0);
    });
}

function calculateBounds(center, sizeInMeters) {
    const earthRadius = 6378137; 
    const lat = center.lat();
    const lng = center.lng();
    const latOffset = (sizeInMeters / earthRadius) * (180 / Math.PI);
    const lngOffset = (sizeInMeters / earthRadius) * (180 / Math.PI) / Math.cos(lat * Math.PI / 180);

    const north = lat + latOffset;
    const south = lat - latOffset;
    const east = lng + lngOffset;
    const west = lng - lngOffset;

    return {selectedBounds: new google.maps.LatLngBounds(
        new google.maps.LatLng(south, west),
        new google.maps.LatLng(north, east)
    ), wideBounds: new google.maps.LatLngBounds(
        new google.maps.LatLng(south - 0.5*latOffset, west - 0.5*lngOffset),
        new google.maps.LatLng(north + 0.5*latOffset, east + 0.5*lngOffset)
    )};
}

function sendToServer(bounds) {
    const north = bounds.getNorthEast().lat();
    const south = bounds.getSouthWest().lat();
    const east = bounds.getNorthEast().lng();
    const west = bounds.getSouthWest().lng();

    const staticMapUrl = 'https://maps.googleapis.com/maps/api/staticmap?' + 
        'center=' + ((north + south) / 2) + ',' + ((east + west) / 2) +
        '&zoom=20' +
        '&size=400x400' +
        '&maptype=satellite' +
        '&key=' + maps_api_key;

    // Show the spinner while waiting for the response
    spinner.style.display = 'block'; // Show the spinner

    fetch(staticMapUrl)
        .then(response => response.blob())
        .then(imageBlob => {
            const reader = new FileReader();
            reader.onloadend = function () {
                const base64Image = reader.result.split(',')[1];
                const data = {
                    north: north,
                    south: south,
                    east: east,
                    west: west,
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

function detectPanels() {
    if (bounds) {
        storedBounds = bounds
        sendToServer(bounds.selectedBounds);
    } else {
        console.log("No bounding box selected.");
        alert("Please click on the map to select an area first.");
    }
}

function displayResponse(responseData) {
    storedResponseDate = responseData
    const responseContent = document.getElementById("responseContent");
    const resultContent = document.getElementById("resultContent");
    
    const feedbackContainer = document.getElementById("feedbackContainer");

    // Clear previous content before displaying new results
    resultContent.innerHTML = '';

    // Create a table to display the JSON data
    const table = document.createElement("table");
    table.style.width = "100%";
    table.setAttribute("border", "1");
    
    // Create table header
    const headerRow = document.createElement("tr");
    const headerKey = document.createElement("th");
    headerKey.textContent = "Model";
    const headerValue = document.createElement("th");
    headerValue.textContent = "Probability";
    headerRow.appendChild(headerKey);
    headerRow.appendChild(headerValue);
    table.appendChild(headerRow);

    // Iterate through responseData and create table rows
    probabilities = responseData.probabilities
    metamodel_probability = responseData.metamodel_probability
    if (metamodel_probability > 0.5){
        new google.maps.Rectangle({
            bounds: storedBounds.selectedBounds,
            editable: false,
            draggable: false,
            map: map,
            strokeColor: '#00FF00', // Change border color to green
            fillColor: '#00FF00'    // Change fill color to green
        });
        selectedArea.setMap(null);
    } else{
        new google.maps.Rectangle({
            bounds: storedBounds.selectedBounds,
            editable: false,
            draggable: false,
            map: map,
            strokeColor: '#FF0000', // Change border color to red
            fillColor: '#FF0000'    // Change fill color to red
        });
        selectedArea.setMap(null);
    }
    map.fitBounds(storedBounds.wideBounds);
    storedBounds = null
    for (const key in probabilities) {
        if (probabilities.hasOwnProperty(key)) {
            const row = document.createElement("tr");
            const cellKey = document.createElement("td");
            cellKey.textContent = key;
            const cellValue = document.createElement("td");
            let value = probabilities[key];
            if (typeof value === 'number') {
                value = value.toFixed(3); // Format the number to 3 decimal places
            }
            cellValue.textContent = value;
            row.appendChild(cellKey);
            row.appendChild(cellValue);
            table.appendChild(row);
        }
    }

    // Append the table to the responseContent div
    resultContent.appendChild(table);

    responseContent.style.display = 'block';

    feedbackContainer.style.display = 'block';
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

function sendFeedback(val){
    document.getElementById("feedbackContainer").style.display = 'none';
    storedResponseDate.feedback = val;
    fetch('/store-feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrf_token  // Make sure to send the CSRF token
        },
        body: JSON.stringify(storedResponseDate)
    }).catch(err => console.error('Request failed', err));
}

window.onload = function() {
    initMap();
};
