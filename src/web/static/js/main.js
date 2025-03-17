/**
 * Fonctions utilitaires pour l'application PokéFight
 */

// Configuration
const API_URL = 'http://localhost:5001/api';

// Formatage des types de Pokémon
function formatType(type) {
    if (!type || type === 'Unknown' || type === 'none') return '';
    return `<span class="type-badge type-${type.toLowerCase()}">${type}</span>`;
}

// Formatage des probabilités
function formatProbability(probability) {
    const percentage = Math.round(probability * 100);
    let colorClass = 'probability-medium';
    
    if (percentage >= 70) {
        colorClass = 'probability-high';
    } else if (percentage < 50) {
        colorClass = 'probability-low';
    }
    
    return `<span class="${colorClass}">${percentage}%</span>`;
}

// Afficher/masquer les messages d'erreur
function showError(message, elementId = 'errorMessage') {
    const errorElement = document.getElementById(elementId);
    if (errorElement) {
        errorElement.classList.remove('hidden');
        const messageElement = errorElement.querySelector('p') || errorElement;
        messageElement.textContent = message;
    }
}

function hideError(elementId = 'errorMessage') {
    const errorElement = document.getElementById(elementId);
    if (errorElement) {
        errorElement.classList.add('hidden');
    }
}

// Gestion de l'état de chargement
function setLoading(isLoading, elementId = 'loadingResults') {
    const loadingElement = document.getElementById(elementId);
    if (loadingElement) {
        if (isLoading) {
            loadingElement.classList.remove('hidden');
        } else {
            loadingElement.classList.add('hidden');
        }
    }
}

// Création d'une carte Pokémon
function createPokemonCard(pokemon, onClick = null, isSelected = false) {
    const card = document.createElement('div');
    card.className = `pokemon-card p-2 border rounded text-center cursor-pointer transition-all ${isSelected ? 'bg-blue-100 border-blue-500' : 'hover:bg-gray-100'}`;
    card.dataset.pokemon = pokemon;
    
    // Nettoyer le nom pour l'URL de l'image
    const cleanName = pokemon.toLowerCase().replace(/[^a-z0-9]/g, '');
    
    card.innerHTML = `
        <img src="https://img.pokemondb.net/artwork/large/${cleanName}.jpg" 
             alt="${pokemon}" 
             class="w-full h-24 object-contain mx-auto"
             onerror="this.onerror=null; this.src='https://via.placeholder.com/96?text=?';">
        <div class="mt-2 font-medium">${pokemon}</div>
    `;
    
    if (onClick) {
        card.addEventListener('click', () => onClick(pokemon));
    }
    
    return card;
}

// Extraction des paramètres d'URL
function getUrlParams() {
    const params = {};
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    
    for (const [key, value] of urlParams.entries()) {
        params[key] = value;
    }
    
    return params;
}

// Fonction pour charger automatiquement les données depuis l'URL
function autoLoadFromUrl(formId, submitButtonId, paramMap = {}) {
    const params = getUrlParams();
    const form = document.getElementById(formId);
    const submitButton = document.getElementById(submitButtonId);
    
    if (!form || !submitButton) return false;
    
    let hasParams = false;
    
    // Remplir le formulaire avec les paramètres d'URL
    for (const [paramName, formField] of Object.entries(paramMap)) {
        if (params[paramName]) {
            const field = form.querySelector(`[name="${formField}"], #${formField}`);
            if (field) {
                field.value = params[paramName];
                hasParams = true;
            }
        }
    }
    
    // Soumettre automatiquement le formulaire si des paramètres ont été trouvés
    if (hasParams && submitButton) {
        setTimeout(() => {
            submitButton.click();
        }, 500);
        return true;
    }
    
    return false;
}

// Fonction pour afficher le spinner de chargement
function showLoading() {
    const loadingSpinner = document.getElementById('loadingSpinner');
    if (loadingSpinner) {
        loadingSpinner.classList.remove('hidden');
    }
}

// Fonction pour masquer le spinner de chargement
function hideLoading() {
    const loadingSpinner = document.getElementById('loadingSpinner');
    if (loadingSpinner) {
        loadingSpinner.classList.add('hidden');
    }
}

// Fonction pour créer un badge de type
function createTypeBadge(type) {
    if (!type) return '';
    
    const badge = document.createElement('span');
    badge.className = `type-badge type-${type.toLowerCase()}`;
    badge.textContent = type;
    
    return badge;
}

// Fonction pour créer un badge de probabilité
function createProbabilityBadge(probability) {
    const probValue = parseFloat(probability);
    let className = 'badge badge-yellow';
    
    if (probValue >= 70) {
        className = 'badge badge-green';
    } else if (probValue < 50) {
        className = 'badge badge-red';
    }
    
    const badge = document.createElement('span');
    badge.className = className;
    badge.textContent = `${probValue}%`;
    
    return badge;
}

// Fonction pour créer un tooltip
function createTooltip(text, tooltipText) {
    const tooltip = document.createElement('span');
    tooltip.className = 'tooltip';
    tooltip.textContent = text;
    
    const tooltipTextElement = document.createElement('span');
    tooltipTextElement.className = 'tooltip-text';
    tooltipTextElement.textContent = tooltipText;
    
    tooltip.appendChild(tooltipTextElement);
    
    return tooltip;
}

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
    // Toggle mobile menu
    const menuButton = document.getElementById('menuButton');
    if (menuButton) {
        menuButton.addEventListener('click', function() {
            const menu = document.getElementById('mobileMenu');
            menu.classList.toggle('hidden');
        });
    }
    
    // Vérifier si nous avons des paramètres dans l'URL
    const params = getUrlParams();
    
    // Si nous sommes sur la page de recherche et qu'un Pokémon est spécifié
    if (window.location.pathname === '/search' && params.pokemon) {
        const pokemonSelect = document.getElementById('pokemonSelect');
        if (pokemonSelect) {
            // Attendre que la liste des Pokémon soit chargée
            const checkSelectInterval = setInterval(() => {
                if (pokemonSelect.options.length > 1) {
                    clearInterval(checkSelectInterval);
                    
                    // Sélectionner le Pokémon spécifié
                    for (let i = 0; i < pokemonSelect.options.length; i++) {
                        if (pokemonSelect.options[i].value === params.pokemon) {
                            pokemonSelect.selectedIndex = i;
                            
                            // Soumettre le formulaire automatiquement
                            document.getElementById('searchForm').dispatchEvent(new Event('submit'));
                            break;
                        }
                    }
                }
            }, 100);
        }
    }
    
    // Si nous sommes sur la page de comparaison et que deux Pokémon sont spécifiés
    if (window.location.pathname === '/compare' && params.pokemon1 && params.pokemon2) {
        const pokemon1Select = document.getElementById('pokemon1Select');
        const pokemon2Select = document.getElementById('pokemon2Select');
        
        if (pokemon1Select && pokemon2Select) {
            // Attendre que les listes de Pokémon soient chargées
            const checkSelectInterval = setInterval(() => {
                if (pokemon1Select.options.length > 1 && pokemon2Select.options.length > 1) {
                    clearInterval(checkSelectInterval);
                    
                    // Sélectionner les Pokémon spécifiés
                    for (let i = 0; i < pokemon1Select.options.length; i++) {
                        if (pokemon1Select.options[i].value === params.pokemon1) {
                            pokemon1Select.selectedIndex = i;
                            break;
                        }
                    }
                    
                    for (let i = 0; i < pokemon2Select.options.length; i++) {
                        if (pokemon2Select.options[i].value === params.pokemon2) {
                            pokemon2Select.selectedIndex = i;
                            break;
                        }
                    }
                    
                    // Soumettre le formulaire automatiquement
                    document.getElementById('compareForm').dispatchEvent(new Event('submit'));
                }
            }, 100);
        }
    }
}); 