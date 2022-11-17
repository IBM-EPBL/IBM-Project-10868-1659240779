const fileUpload = document.getElementById("upload");
const imageContainer = document.getElementById("image-container");
const imagePreview = document.getElementById("image");
const previewButton = document.getElementById("preview");

previewButton.addEventListener("click", (e) => {
    const [image] = fileUpload.files;
    console.log(fileUpload.files)
    console.log(image)
    if (image) {
        imageContainer.style.display = "block";
        imagePreview.src = URL.createObjectURL(image)
    }
})