// Select elements from the DOM
const wrapper = document.querySelector('.wrapper');
const registerLink = document.querySelector('.register-link');
const loginLink = document.querySelector('.login-link');

// Function for showing the register form
registerLink.onclick = () => {
    wrapper.classList.add('active');
};

// Function for showing the login form
loginLink.onclick = () => {
    wrapper.classList.remove('active');
};




const list = document.querySelectorAll('.list');
function activeLink(){
    list.forEach((item)=>
    item.classList.remove('active'));
    this.classList.add('active');
}
list.forEach((item) =>
item.addEventListener('click',activeLink));