(async () => {
  const res = await fetch("https://foodish-api.herokuapp.com/api/");
  const body = document.querySelector("body");

  const resJSON = await res.json();
  body.style.backgroundImage = `url(${resJSON.image})`;
})();
