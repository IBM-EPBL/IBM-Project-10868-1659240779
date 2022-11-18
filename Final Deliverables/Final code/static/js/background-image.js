(async () => {
  const res = await fetch("https://foodish-api.herokuapp.com/api/");
  const bg = document.querySelector("body");

  const resJSON = await res.json();
  bg.style.backgroundImage = `url(${resJSON.image})`;
})();
