window.onload = function () {
    var supportedFlag = $.keyframe.isSupported();

    function getRandomInt(max) {
        return Math.floor(Math.random() * max);
    }

    var url_start = location.href
    let chara = Array("nduel", 'nbrawl', 'ncitizen', "nvillain", "nseer", "npressured");
    var url = new URL(url_start);
    var nduel = url.searchParams.get(chara[0]);
    var nbrawl = url.searchParams.get(chara[1]);
    var ncitizen = url.searchParams.get(chara[2]);
    var nvillain = url.searchParams.get(chara[3]);
    var nseer = url.searchParams.get(chara[4]);
    var npressured = url.searchParams.get(chara[5]);

    var container = document.getElementById("city");
    if (nduel !== null) {
        for (var i = 0; i < nduel; i++) {
            container.innerHTML += "<div class='person warrior' id='duel" + i + "'>💂</div>";
            element = document.getElementById('duel' + i);
            var x = getRandomInt(700);
            var y = getRandomInt(500);
            element.style.webkitTransform = "translate(" + x + "px, " + y + "px) rotateX(-90deg) rotateY(180deg)";
            var tx25 = getRandomInt(600); var tx50 = getRandomInt(600);
            var ty25 = getRandomInt(600); var ty50 = getRandomInt(600);
            var tx75 = getRandomInt(600);
            var ty75 = getRandomInt(600);

            $.keyframe.define([{
                name: 'key_duel' + i,
                '25%': { "transform": "translate(" + tx25 + "px, " + ty25 + "px) rotateX(-90deg) rotateY(180deg)" },
                '50%': { "transform": "translate(" + tx50 + "px, " + ty50 + "px) rotateX(-90deg) rotateY(0deg)" },
                '75%': { "transform": "translate(" + tx75 + "px, " + ty75 + "px)rotateX(-90deg) rotateY(0deg)" },
                '100%': {
                    "transform": "translate(" + x + "px, " + y + "px) rotateX(-90deg) rotateY(-180deg)"
                },
            }]);

            $("#duel" + i).playKeyframe({
                name: 'key_duel' + i, // name of the keyframe you want to bind to the selected element
                duration: '35s', // [optional, default: 0, in ms] how long you want it to last in milliseconds
                timingFunction: 'linear', // [optional, default: ease] specifies the speed curve of the animation
                delay: '0s', //[optional, default: 0s]  how long you want to wait before the animation starts
                iterationCount: 'infinite', //[optional, default:1]  how many times you want the animation to repeat
                direction: 'normal', //[optional, default: 'normal']  which direction you want the frames to flow
                fillMode: 'forwards', //[optional, default: 'forward']  how to apply the styles outside the animation time, default value is forwards
                complete: function () { } //[optional] Function fired after the animation is complete. If repeat is infinite, the function will be fired every time the animation is restarted.
            });
        }
    };
    if (nbrawl !== null) {
        for (var i = 0; i < nbrawl; i++) {
            container.innerHTML += "<div class='person warrior' id='brawl" + i + "'>💂</div>";
            element = document.getElementById('brawl' + i);
            var x = getRandomInt(700);
            var y = getRandomInt(500);
            element.style.webkitTransform = "translate(" + x + "px, " + y + "px) rotateX(-90deg) rotateY(180deg)";
            var tx25 = getRandomInt(600); var tx50 = getRandomInt(600);
            var ty25 = getRandomInt(600); var ty50 = getRandomInt(600);
            var tx75 = getRandomInt(600);
            var ty75 = getRandomInt(600);
            $.keyframe.define([{
                name: 'key_brawl' + i,
                '25%': { "transform": "translate(" + tx25 + "px, " + ty25 + "px) rotateX(-90deg) rotateY(180deg)" },
                '50%': { "transform": "translate(" + tx50 + "px, " + ty50 + "px) rotateX(-90deg) rotateY(0deg)" },
                '75%': { "transform": "translate(" + tx75 + "px, " + ty75 + "px)rotateX(-90deg) rotateY(0deg)" },
                '100%': {
                    "transform": "translate(" + x + "px, " + y + "px) rotateX(-90deg) rotateY(-180deg)"
                },
            }]);
            $("#brawl" + i).playKeyframe({
                name: 'key_brawl' + i, // name of the keyframe you want to bind to the selected element
                duration: '40s', // [optional, default: 0, in ms] how long you want it to last in milliseconds
                timingFunction: 'linear', // [optional, default: ease] specifies the speed curve of the animation
                iterationCount: 'infinite', //[optional, default:1]  how many times you want the animation to repeat
            });
        }
    };

    const citizens = ["👩‍👩‍👧", "👨‍👨‍👦"];

    if (ncitizen !== null) {
        for (var i = 0; i < ncitizen; i++) {
            const random = Math.floor(Math.random() * citizens.length);
            var chosen = citizens[random];
            if (chosen === "👩‍👩‍👧") {
                container.innerHTML += "<div class='person citizen-1' id='citizen" + i + "'>👩‍👩‍👧</div>";
            }
            else {
                container.innerHTML += "<div class='person citizen-2' id='citizen" + i + "'>👨‍👨‍👦</div>";
            }
            element = document.getElementById('citizen' + i);
            var x = getRandomInt(700);
            var y = getRandomInt(500);
            element.style.webkitTransform = "translate(" + x + "px, " + y + "px) rotateX(-90deg) rotateY(180deg)";
            var tx25 = getRandomInt(600); var tx50 = getRandomInt(600);
            var ty25 = getRandomInt(600); var ty50 = getRandomInt(600);
            var tx75 = getRandomInt(600);
            var ty75 = getRandomInt(600);
            $.keyframe.define([{
                name: 'key_citizen' + i,
                '25%': { "transform": "translate(" + tx25 + "px, " + ty25 + "px) rotateX(-90deg) rotateY(180deg)" },
                '50%': { "transform": "translate(" + tx50 + "px, " + ty50 + "px) rotateX(-90deg) rotateY(0deg)" },
                '75%': { "transform": "translate(" + tx75 + "px, " + ty75 + "px)rotateX(-90deg) rotateY(0deg)" },
                '100%': {
                    "transform": "translate(" + x + "px, " + y + "px) rotateX(-90deg) rotateY(-180deg)"
                },
            }]);
            $("#citizen" + i).playKeyframe({
                name: 'key_citizen' + i, // name of the keyframe you want to bind to the selected element
                duration: '45s', // [optional, default: 0, in ms] how long you want it to last in milliseconds
                timingFunction: 'linear', // [optional, default: ease] specifies the speed curve of the animation
                iterationCount: 'infinite', //[optional, default:1]  how many times you want the animation to repeat
            });
        }
    };

    const villains = ["🦹", "👹"];

    if (nvillain !== null) {
        for (var i = 0; i < nvillain; i++) {
            const random = Math.floor(Math.random() * villains.length);
            var chosen = villains[random];
            if (chosen === "🦹") {
                container.innerHTML += "<div class='person villain-1' id='villain" + i + "'>🦹</div>";
            }
            else {
                container.innerHTML += "<div class='person villain-2' id='villain" + i + "'>👹</div>";
            }
            element = document.getElementById('villain' + i);
            var x = getRandomInt(700);
            var y = getRandomInt(500);
            element.style.webkitTransform = "translate(" + x + "px, " + y + "px) rotateX(-90deg) rotateY(180deg)";
            var tx25 = getRandomInt(600); var tx50 = getRandomInt(600);
            var ty25 = getRandomInt(600); var ty50 = getRandomInt(600);
            var tx75 = getRandomInt(600);
            var ty75 = getRandomInt(600);
            $.keyframe.define([{
                name: 'key_villain' + i,
                '25%': { "transform": "translate(" + tx25 + "px, " + ty25 + "px) rotateX(-90deg) rotateY(180deg)" },
                '50%': { "transform": "translate(" + tx50 + "px, " + ty50 + "px) rotateX(-90deg) rotateY(0deg)" },
                '75%': { "transform": "translate(" + tx75 + "px, " + ty75 + "px)rotateX(-90deg) rotateY(0deg)" },
                '100%': {
                    "transform": "translate(" + x + "px, " + y + "px) rotateX(-90deg) rotateY(-180deg)"
                },
            }]);
            $("#villain" + i).playKeyframe({
                name: 'key_villain' + i, // name of the keyframe you want to bind to the selected element
                duration: '30s', // [optional, default: 0, in ms] how long you want it to last in milliseconds
                timingFunction: 'linear', // [optional, default: ease] specifies the speed curve of the animation
                iterationCount: 'infinite', //[optional, default:1]  how many times you want the animation to repeat
            });
        }
    };

    if (nseer !== null) {
        for (var i = 0; i < nseer; i++) {
            container.innerHTML += "<div class='person wizard' id='seer" + i + "'>🧙</div>";
            element = document.getElementById('seer' + i);
            var x = getRandomInt(700);
            var y = getRandomInt(500);
            element.style.webkitTransform = "translate(" + x + "px, " + y + "px) rotateX(-90deg) rotateY(180deg)";
            var tx25 = getRandomInt(600); var tx50 = getRandomInt(600);
            var ty25 = getRandomInt(600); var ty50 = getRandomInt(600);
            var tx75 = getRandomInt(600);
            var ty75 = getRandomInt(600);
            $.keyframe.define([{
                name: 'key_seer' + i,
                '25%': { "transform": "translate(" + tx25 + "px, " + ty25 + "px) rotateX(-90deg) rotateY(180deg)" },
                '50%': { "transform": "translate(" + tx50 + "px, " + ty50 + "px) rotateX(-90deg) rotateY(0deg)" },
                '75%': { "transform": "translate(" + tx75 + "px, " + ty75 + "px)rotateX(-90deg) rotateY(0deg)" },
                '100%': {
                    "transform": "translate(" + x + "px, " + y + "px) rotateX(-90deg) rotateY(-180deg)"
                },
            }]);
            $("#seer" + i).playKeyframe({
                name: 'key_seer' + i, // name of the keyframe you want to bind to the selected element
                duration: '30s', // [optional, default: 0, in ms] how long you want it to last in milliseconds
                timingFunction: 'linear', // [optional, default: ease] specifies the speed curve of the animation
                iterationCount: 'infinite', //[optional, default:1]  how many times you want the animation to repeat
            });
        }
    };
    if (npressured !== null) {
        for (var i = 0; i < npressured; i++) {
            container.innerHTML += "<div class='person wizard' id='pressured" + i + "'>🧙</div>";
            element = document.getElementById('pressured' + i);
            var x = getRandomInt(700);
            var y = getRandomInt(500);
            element.style.webkitTransform = "translate(" + x + "px, " + y + "px) rotateX(-90deg) rotateY(180deg)";
            var tx25 = getRandomInt(600); var tx50 = getRandomInt(600);
            var ty25 = getRandomInt(600); var ty50 = getRandomInt(600);
            var tx75 = getRandomInt(600);
            var ty75 = getRandomInt(600);
            $.keyframe.define([{
                name: 'key_pressured' + i,
                '25%': { "transform": "translate(" + tx25 + "px, " + ty25 + "px) rotateX(-90deg) rotateY(180deg)" },
                '50%': { "transform": "translate(" + tx50 + "px, " + ty50 + "px) rotateX(-90deg) rotateY(0deg)" },
                '75%': { "transform": "translate(" + tx75 + "px, " + ty75 + "px)rotateX(-90deg) rotateY(0deg)" },
                '100%': {
                    "transform": "translate(" + x + "px, " + y + "px) rotateX(-90deg) rotateY(-180deg)"
                },
            }]);
            $("#pressured" + i).playKeyframe({
                name: 'key_pressured' + i, // name of the keyframe you want to bind to the selected element
                duration: '20s', // [optional, default: 0, in ms] how long you want it to last in milliseconds
                timingFunction: 'linear', // [optional, default: ease] specifies the speed curve of the animation
                iterationCount: 'infinite', //[optional, default:1]  how many times you want the animation to repeat
            });
        }
    };
    const trees = ["🌲", "🌳"];
    for (j = 0; j < 20; j++) {
        var nb = getRandomInt(7);
        const random = Math.floor(Math.random() * trees.length);
        var chosen = trees[random];
        if (chosen === "🌲") {
            container.innerHTML += "<div class='tree tree-" + nb + "' id='tree" + j + "'>🌲</div>";
        }
        else {
            container.innerHTML += "<div class='tree tree-" + nb + "' id='tree" + j + "'>🌳</div>";
        }
        element = document.getElementById('tree' + j);

        var x = getRandomInt(700);
        var y = getRandomInt(500);
        element.style.webkitTransform = "translate(" + x + "px, " + y + "px) rotateX(-90deg) rotateY(calc(var(--rotate-y)*-1))";
    };

    const houses = ["🏰", "🛖"];
    const degrees = [90, 180];
    for (j = 0; j < 10; j++) {
        var random = Math.floor(Math.random() * houses.length);
        var chosen = houses[random];
        if (chosen === "🏰") {
            container.innerHTML += "<div class='house house-1' id='house" + j + "'>🏰</div>";
        }
        else {
            container.innerHTML += "<div class='house house-1' id='house" + j + "'>🛖</div>";
        }
        element = document.getElementById('house' + j);
        var x = getRandomInt(500);
        var y = getRandomInt(400);
        var rotx = -90;
        var random = Math.floor(Math.random() * houses.length);
        var roty = degrees[random];
        element.style.webkitTransform = "translate(" + x + "px, " + y + "px) rotateX(" + rotx + "deg) rotateY(" + roty + "deg)";
    };
}

