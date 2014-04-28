$( document ).ready(function() {
    console.log( "ready!" );
    
    var greetings = Array("Hello, my name is",
                      "Bonjour, je m'appelle",
                      "Hola, me llamo",
                      "Здраствуите, меня завут",
                      "Hallo, ich weisse")
    
    $('#hello').click(function() {
        $(this).text(greetings[Math.floor(Math.random() * greetings.length)]);
    });
});


                      

