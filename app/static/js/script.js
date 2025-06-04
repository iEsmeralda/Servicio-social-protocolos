//Funcionamiento del botón
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