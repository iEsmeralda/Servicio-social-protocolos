// Funcionamiento del botón de búscar
document.getElementById("btnSistema").addEventListener("click", function(event) {
  event.preventDefault(); // Evita la recarga automática de la página

  var divRecomendaciones = document.getElementById("Recomendaciones");
  var consulta = document.getElementById("consulta");

  var query = consulta.value;

  // Validar que se haya ingresado un valor en el campo de la consulta
  if (!query.trim()) {
    alert("Por favor, ingresa un texto del tema de interés para encontrar protocolos relacionados.");
    consulta.value = "";
    return; // Detener la ejecución si no se cumple la validación
  }

  fetch('/resultadosProtocolos', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: 'consulta=' + query
  })
  .then(response => response.text())
  .then(data => {
    // Mostrar las recomendaciones en la página web
    divRecomendaciones.innerHTML = data; // Actualizar el contenido del div con los datos recibidos
    console.log(data);
  });
});

// Funcionamiento del form de los datos para filtrado
document.getElementById("formFiltros").addEventListener("submit", function(event) {
  event.preventDefault();
  
  var divRecomendaciones = document.getElementById("Recomendaciones");
  var consulta = document.getElementById("consulta");
  var profesor = document.getElementById("profe");
  var anio = document.getElementById("anio");

  var query = consulta.value;
  var nombreProfesor = profesor.value;
  var anioTT = anio.value;

  // Validar que se haya ingresado un valor en el campo de la consulta, profesor y año del TT
  if (!query.trim() && (!nombreProfesor.trim() || !anioTT.trim())) {
    alert("Por favor, ingresa un texto del tema de interés así como el campo que desees para el filtro.");
    consulta.value = "";
    return; // Detener la ejecución si no se cumple la validación
  }

  // Crear los parámetros codificados para enviar por POST
  var params = new URLSearchParams();
  params.append('consulta', query);
  params.append('profesor', nombreProfesor);
  params.append('anio', anioTT);

  fetch('/resultadosProtocolosConFiltro', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: params.toString()
  })
  .then(response => response.text())
  .then(data => {
    // Mostrar las recomendaciones en la página web
    divRecomendaciones.innerHTML = data; // Actualizar el contenido del div con los datos recibidos
    console.log(data);
  });
});

// // // Funcionamiento del botón de filtros
// // document.getElementById("btnFiltro").addEventListener("click", function(event) {
// //   event.preventDefault(); // Evita la recarga automática de la página

  
// // });