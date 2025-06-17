document.getElementById("btnSistema").addEventListener("click", function(event) {
  event.preventDefault();

  var divRecomendaciones = document.getElementById("Recomendaciones");
  var consulta = document.getElementById("consulta");
  var loader = document.getElementById("loading");

  var query = consulta.value;

  if (!query.trim()) {
    alert("Por favor, ingresa un texto del tema de interés para encontrar protocolos relacionados.");
    consulta.value = "";
    return;
  }

  // Mostrar loader
  loader.style.display = "block";
  divRecomendaciones.innerHTML = "";

  fetch('/resultadosProtocolos', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: 'consulta=' + query
  })
  .then(response => response.text())
  .then(data => {
    divRecomendaciones.innerHTML = data;
  })
  .finally(() => {
    // Ocultar loader
    loader.style.display = "none";
  });
});


// Funcionamiento del form de los datos para filtrado
document.getElementById("formFiltros").addEventListener("submit", function(event) {
  event.preventDefault();

  var divRecomendaciones = document.getElementById("Recomendaciones");
  var consulta = document.getElementById("consulta");
  var profesor = document.getElementById("profe");
  var anio = document.getElementById("anio");
  var loader = document.getElementById("loading");

  var query = consulta.value;
  var nombreProfesor = profesor.value;
  var anioTT = anio.value;

  if (!query.trim() && (!nombreProfesor.trim() || !anioTT.trim())) {
    alert("Por favor, ingresa un texto del tema de interés así como el campo que desees para el filtro.");
    consulta.value = "";
    return;
  }

  var params = new URLSearchParams();
  params.append('consulta', query);
  params.append('profesor', nombreProfesor);
  params.append('anio', anioTT);

  // Mostrar loader
  loader.style.display = "block";
  divRecomendaciones.innerHTML = "";

  fetch('/resultadosProtocolosConFiltro', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: params.toString()
  })
  .then(response => response.text())
  .then(data => {
    divRecomendaciones.innerHTML = data;
  })
  .finally(() => {
    // Ocultar loader
    loader.style.display = "none";
  });
});


// // // Funcionamiento del botón de filtros
// // document.getElementById("btnFiltro").addEventListener("click", function(event) {
// //   event.preventDefault(); // Evita la recarga automática de la página

  
// // });