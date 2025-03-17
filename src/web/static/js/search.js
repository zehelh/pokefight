// Fonction pour rechercher les counters
function searchCounters() {
    const pokemonName = document.getElementById('pokemon-search').value.trim();
    if (!pokemonName) {
        showMessage("Veuillez entrer un nom de Pokémon", "error");
        return;
    }

    // Afficher un loader
    document.getElementById('results-container').innerHTML = '<div class="loader"></div>';
    document.getElementById('results-container').style.display = 'block';

    // Faire la requête à l'API
    fetch(`/api/counters/${pokemonName}?top_k=5`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            displayCountersResults(pokemonName, data);
        })
        .catch(error => {
            console.error('Erreur lors de la recherche:', error);
            document.getElementById('results-container').innerHTML = `
                <div class="error-message">
                    <p>Erreur lors de la recherche des contre-Pokémon.</p>
                    <p>Détails: ${error.message}</p>
                </div>
            `;
        });
}

// Fonction pour afficher les résultats de la recherche
function displayCountersResults(targetPokemon, data) {
    const resultsContainer = document.getElementById('results-container');
    
    // Si la réponse est un objet avec une propriété 'counters'
    let counters = data;
    if (data && typeof data === 'object' && data.counters) {
        counters = data.counters;
        // Mettre à jour targetPokemon si disponible dans la réponse
        if (data.target_pokemon) {
            targetPokemon = data.target_pokemon;
        }
    }
    
    // Vérifiez si counters existe et n'est pas vide
    if (!counters || counters.length === 0) {
        resultsContainer.innerHTML = `
            <div class="no-results">
                <h3>Aucun contre pour ${targetPokemon}</h3>
                <p>Aucun contre-Pokémon n'a été trouvé pour ${targetPokemon}.</p>
            </div>
        `;
        return;
    }
    
    // Créer l'en-tête des résultats
    let resultsHTML = `
        <h2>Meilleurs contre-Pokémon pour ${targetPokemon}</h2>
    `;
    
    // Créer une grille pour les cartes de counter
    resultsHTML += '<div class="counter-cards">';
    
    // Ajouter chaque counter
    counters.forEach(counter => {
        resultsHTML += createCounterCard(counter, targetPokemon);
    });
    
    resultsHTML += '</div>';
    
    // Mettre à jour le contenu
    resultsContainer.innerHTML = resultsHTML;
}

// Fonction pour créer une carte de counter
function createCounterCard(counter, targetPokemon) {
    // Déterminer si nous utilisons une probabilité originale ou ajustée
    const hasOriginalProb = counter.hasOwnProperty('original_win_probability');
    const winProb = counter.win_probability;
    const originalProb = hasOriginalProb ? counter.original_win_probability : winProb;
    
    // Déterminer la couleur en fonction de la probabilité
    let colorClass = '';
    if (winProb >= 0.85) {
        colorClass = 'excellent';
    } else if (winProb >= 0.7) {
        colorClass = 'good';
    } else if (winProb >= 0.55) {
        colorClass = 'average';
    } else {
        colorClass = 'poor';
    }
    
    // Formater les probabilités en pourcentage
    const winProbPercent = Math.round(winProb * 100);
    const originalProbPercent = Math.round(originalProb * 100);
    
    // Vérifier si la source inclut une mention d'erreur ou de secours
    const isBackupData = counter.source && counter.source.includes('Secours');
    
    // Créer la liste des raisons
    let reasonsHTML = '<ul class="reasons-list">';
    counter.reasons.forEach(reason => {
        reasonsHTML += `<li>${reason}</li>`;
    });
    reasonsHTML += '</ul>';
    
    // Construire la carte HTML
    return `
        <div class="counter-card ${isBackupData ? 'backup-data' : ''}">
            <h3>${counter.pokemon}</h3>
            <div class="pokemon-image">
                <img src="https://img.pokemondb.net/artwork/large/${counter.pokemon.toLowerCase().replace(/\s+/g, '')}.jpg" 
                     onerror="this.src='https://via.placeholder.com/120x120?text=${counter.pokemon}'" 
                     alt="${counter.pokemon}">
            </div>
            
            <div class="probability-container">
                <div class="circular-progress ${colorClass}">
                    <div class="progress-value">${winProbPercent}%</div>
                </div>
                <div class="original-score">Score initial: ${originalProbPercent}%</div>
            </div>
            
            <h4>Avantages:</h4>
            ${reasonsHTML}
            
            <div class="source-info">
                Source: ${counter.source || 'Modèle optimisé'}
            </div>
        </div>
    `;
}

// Fonction pour afficher un message
function showMessage(message, type = 'info') {
    const messageContainer = document.getElementById('message-container');
    if (!messageContainer) return;
    
    const messageElement = document.createElement('div');
    messageElement.className = `message ${type}`;
    messageElement.innerHTML = `
        <span class="message-text">${message}</span>
        <button class="close-btn">&times;</button>
    `;
    
    // Ajouter le message
    messageContainer.appendChild(messageElement);
    
    // Configurer le bouton de fermeture
    const closeBtn = messageElement.querySelector('.close-btn');
    closeBtn.addEventListener('click', () => {
        messageContainer.removeChild(messageElement);
    });
    
    // Disparition automatique après 5 secondes
    setTimeout(() => {
        if (messageElement.parentNode === messageContainer) {
            messageContainer.removeChild(messageElement);
        }
    }, 5000);
}

// Initialiser la page quand elle est chargée
document.addEventListener('DOMContentLoaded', function() {
    console.log("Page search.js chargée");
    
    const searchInput = document.getElementById('pokemon-search');
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchCounters();
            }
        });
        
        // Charger la liste des Pokémon pour l'autocomplétion
        fetch('/api/local/pokemon')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Erreur HTTP: ${response.status}`);
                }
                return response.json();
            })
            .then(pokemonList => {
                if (Array.isArray(pokemonList)) {
                    // Créer un datalist pour l'autocomplétion
                    const datalist = document.createElement('datalist');
                    datalist.id = 'pokemon-list';
                    
                    pokemonList.forEach(pokemon => {
                        const option = document.createElement('option');
                        option.value = pokemon;
                        datalist.appendChild(option);
                    });
                    
                    document.body.appendChild(datalist);
                    searchInput.setAttribute('list', 'pokemon-list');
                    
                    console.log(`Liste d'autocomplétion créée avec ${pokemonList.length} Pokémon`);
                }
            })
            .catch(error => {
                console.error('Erreur lors du chargement de la liste des Pokémon:', error);
            });
    }
}); 