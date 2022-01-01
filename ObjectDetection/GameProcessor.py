import cv2.cv2

from homography import *
from pawns_recognition import *
from cards_recognition import *
from GameBoard import *
from drawing import *

class GameProcessor:
    capEveryFrame = False
    homographymatrixfound = False
    checkInitialPosition = False #TODO

    def __init__(self, img, window_name):
        self.window_name = window_name

        self.height = img.shape[0]
        self.width = img.shape[1]

        self.gameBoard = GameBoard()
        self.pawnsRecognitionHelper = PawnsRecognitionHelper(self.height, self.width, self.gameBoard)
        self.cardsRecognitionHelper = CardsRecognitionHelper(self.height, self.width, self.gameBoard)

        cv2.setMouseCallback(self.window_name, self.ComputeMouseInput)

        self.list_board_coords = []
        self.actionPawnClicked = None
        self.showAlibi = False
        self.isJackSeen = False

    def ComputeFrame(self, img):

        # If we are still waiting for the homography points
        if not self.homographymatrixfound:
            # When we have the 4 coordinates we can compute it
            if len(self.list_board_coords) == 4:
                # When we get the points for homography, set homographymatrix
                self.homographymatrix, self.coordinates = get_homography_matrix(img, np.array(self.list_board_coords), self.width, self.height)

                # Compute coordinates required for pawns and cards recognition helper
                self.cardsRecognitionHelper.GetScreenPortions(img, self.coordinates)
                self.pawnsRecognitionHelper.GetScreenPortion(img, self.coordinates)
                
                self.homographymatrixfound = True
                img = cv2.warpPerspective(img, self.homographymatrix, (img.shape[1], img.shape[0]))

                # Then directly compute the cards and their orientation
                self.cardsRecognitionHelper.ComputeFrame(img)

                if(self.checkInitialPosition):
                    if (self.gameBoard.validateCardsInitialPosition()):
                        self.gameBoard.updatePreviousCards()
                        self.gameBoard.tryUpdateGameStatus(GameStates.GSWaitingActionPawnsThrow)
                    else:
                        print("Le placement initial des cartes n'est pas bon")  # TODO ERROR DISPLAY
                else:
                    self.gameBoard.updatePreviousCards()
                    self.gameBoard.tryUpdateGameStatus(GameStates.GSWaitingActionPawnsThrow)
        else:
            # We need to apply homography here so that the frames are correctly computed
            img = cv2.warpPerspective(img, self.homographymatrix, (img.shape[1], img.shape[0]))

            # If we have a clicked Action Pawn to compute
            if self.actionPawnClicked is not None:
                self.UseActionPawn(img, self.actionPawnClicked)

                # Whether we were able to use the action pawn or not, it is not clicked anymore
                self.actionPawnClicked = None

        return img

    def DrawFrame(self, img):
        modifiedimg = img.copy()

        if not self.homographymatrixfound:
            modifiedimg = drawText(modifiedimg, "Selectionnez les quatre coins des 9 cartes",TextPositions.TPCenter)
            for coord in self.list_board_coords:
                cv2.circle(modifiedimg, coord, 10, (0, 255, 0), -1)
        elif self.gameBoard.getGameStatus() == GameStates.GSWaitingCards:
            modifiedimg = self.pawnsRecognitionHelper.DrawZonesRectangles(modifiedimg, drawOffset=True)
            modifiedimg = drawMultipleLinesOfText(modifiedimg, ["Appuyez sur C pour detecter les cartes"], TextPositions.TPTopL)
        elif self.gameBoard.getGameStatus() == GameStates.GSWaitingActionPawnsThrow:
            modifiedimg = self.pawnsRecognitionHelper.DrawZonesRectangles(modifiedimg, drawOffset=True)
            modifiedimg = self.cardsRecognitionHelper.DrawFrame(modifiedimg)
            modifiedimg = drawMultipleLinesOfText(modifiedimg, ["Appuyez sur P pour detecter les pions", "Ou sur C pour redetecter les cartes"], TextPositions.TPTopL)
        elif self.gameBoard.getGameStatus() == GameStates.GSUsingActionPawns:
            # We ask for player input of show the action played by IA
            if (self.gameBoard.getCurrentPlayer() == "Detective"):
                # If not in the case where we need to show the special alibi msg, we show the basic use action pawn msg
                if not self.showAlibi:
                    modifiedimg = self.cardsRecognitionHelper.DrawFrame(modifiedimg)
                    modifiedimg = self.pawnsRecognitionHelper.DrawFrame(modifiedimg)
                    modifiedimg = drawMultipleLinesOfText(modifiedimg, ["Realisez votre Action puis","Appuyez sur le jeton correspondant","Ou sur P pour redetecter les pions"], TextPositions.TPTopL)
                    modifiedimg = drawPlayerAndTurn(modifiedimg, self.gameBoard.getCurrentPlayer(), self.gameBoard.getTurnCount())
                else :
                    innocentedCard = self.gameBoard.getInnocentedCard()
                    modifiedimg = drawMultipleLinesOfText(modifiedimg, ["La carte alibi tiree est : " + innocentedCard, "Retournez la carte innocente si elle est presente", "Puis appuyez sur espace"], TextPositions.TPTopL)
                    modifiedimg = self.cardsRecognitionHelper.DrawBoxesByName(modifiedimg, [innocentedCard])
                    #modifiedimg = drawPlayerAndTurn(modifiedimg, self.gameBoard.getCurrentPlayer(), self.gameBoard.getTurnCount())

            else:
                modifiedimg = drawPlayerAndTurn(modifiedimg, self.gameBoard.getCurrentPlayer(), self.gameBoard.getTurnCount())
                IAAction = self.gameBoard.getIaAction()
                if IAAction is not None:
                    modifiedimg = self.DrawIAAction(modifiedimg, IAAction)

        elif self.gameBoard.getGameStatus() == GameStates.GSAppealOfWitness:
            if self.isJackSeen:
                jackString = "Jack est en vue des detectives"
            else:
                jackString = "Jack n'est pas en vue des detectives"
            modifiedimg = drawMultipleLinesOfText(modifiedimg, [jackString, "Retournez les cartes innocentees","Puis appuyez sur C"], TextPositions.TPTopL)
            innocentCards = self.gameBoard.getInnocentCards()
            modifiedimg = self.cardsRecognitionHelper.DrawBoxesByName(modifiedimg, innocentCards)

        elif self.gameBoard.getGameStatus() == GameStates.GSGameOver:
            if (self.gameBoard.getDetectiveWins()):
                winnerstring = "Vous avez gagne"
            else:
                winnerstring = "Jack a gagne"
            modifiedimg = drawMultipleLinesOfText(modifiedimg, ["Partie terminee",winnerstring], TextPositions.TPCenter)

        return modifiedimg

    def DrawIAAction(self, img, action):
        actionPawnPlayed = ActionPawns[action[0]]
        # If the action pawn played is regarding detective pawns
        if actionPawnPlayed.value in [0,2,3,4]:
            img = self.pawnsRecognitionHelper.DrawDetectivePawnByName(img, action[1][0])
            img = drawMultipleLinesOfText(img,["Deplacez le jeton entoure de " + str(action[1][1]) + " cases", "Puis appuyez sur espace pour valider"], TextPositions.TPTopL)
        elif (actionPawnPlayed == ActionPawns.APReturn or actionPawnPlayed == ActionPawns.APReturn2):
            img = self.cardsRecognitionHelper.DrawBoxesByIndex(img, [action[1][0]])
            img = drawMultipleLinesOfText(img, ["Tournez le jeton entoure vers : " + action[1][1], "Puis appuyez sur espace pour valider"], TextPositions.TPTopL)
        elif (actionPawnPlayed == ActionPawns.APChangeCard):
            img = self.cardsRecognitionHelper.DrawBoxesByIndex(img, [action[1][0],action[1][1]])
            img = drawMultipleLinesOfText(img, ["Echangez de place les 2 cartes entourees", "Puis appuyez sur espace pour valider"], TextPositions.TPTopL)
        else:
            img = drawMultipleLinesOfText(img, ["Jack a tire une carte Alibi", "Appuyez sur espace pour valider"], TextPositions.TPTopL)

        return img

    def ComputeInputs(self, img):
        continuebool = True

        key = cv2.waitKey(1) & 0xFF

        # Exit if Q or ESC pressed
        if key == ord('q') or key == 27:
            continuebool = False
        # If we press C for detect Cards
        if (key == ord('c')): # or self.capEveryFrame
            # First this is here we check for victory, when we detect the cards after a ManHunt
            if (self.gameBoard.getDetectiveWins() or self.gameBoard.getJackWins()):
                self.gameBoard.tryUpdateGameStatus(GameStates.GSGameOver)

            elif (self.gameBoard.canUpdateGameStatus(GameStates.GSWaitingActionPawnsThrow)):
                self.cardsRecognitionHelper.ComputeFrame(img)

                # If we are doing the card recognition for the first turn we need to check if all the cards are placed correctly
                if (self.gameBoard.getTurnCount() == 1 and self.checkInitialPosition):
                    if (self.gameBoard.validateCardsInitialPosition()):
                        self.gameBoard.updatePreviousCards()
                        self.gameBoard.tryUpdateGameStatus(GameStates.GSWaitingActionPawnsThrow)
                    else:
                        print("Le placement initial des cartes n'est pas bon") #TODO ERROR DISPLAY
                else:
                    self.gameBoard.updatePreviousCards()
                    self.gameBoard.tryUpdateGameStatus(GameStates.GSWaitingActionPawnsThrow)


        # If we press P for detect Pawns
        if (key == ord('p')): # or self.capEveryFrame

            if (self.gameBoard.canUpdateGameStatus(GameStates.GSUsingActionPawns)):
                self.pawnsRecognitionHelper.ComputeFrame(img)

                # If we are doing the pawns recognition for the first turn we need to check if all the pawns are placed correctly
                if (self.gameBoard.getTurnCount() == 1 and self.checkInitialPosition):
                    if (self.gameBoard.validatePawnsInitialPosition()):
                        self.gameBoard.updatePreviousPawnsState()
                        self.gameBoard.tryUpdateGameStatus(GameStates.GSUsingActionPawns)
                    else:
                        print("Le placement initial des pions n'est pas bon")  # TODO ERROR DISPLAY
                else:
                    self.gameBoard.updatePreviousPawnsState()
                    self.gameBoard.tryUpdateGameStatus(GameStates.GSUsingActionPawns)


        # If we press space to validate IA Action or Alibi Card show
        if (key == 32): # or self.capEveryFrame
            if (self.showAlibi):
                # We finish the processing of our alibi card here
                self.UseActionPawn(img, ActionPawns.APAlibi, IATurn = False, EndAlibiProcessing = True)

                self.showAlibi = False

            elif (self.gameBoard.currentPlayer == "Jack"):
                IAAction = ActionPawns[self.gameBoard.getIaAction()[0]]
                self.UseActionPawn(img, IAAction, IATurn=True)

        return continuebool

    def UseActionPawn(self, img, actionPawn, IATurn = False, EndAlibiProcessing = False):
        # We skip the first part in case it is an alibi card that we need to validate
        if EndAlibiProcessing == False:
            # Different actions depending on the AP clicked
            if (actionPawn.value in [0,2,3,4]):
                self.pawnsRecognitionHelper.ComputeDetectivePawns(img)

                print("Previous : \n", self.gameBoard.getPreviousDetectivePawns(), "\nCurrent: \n", self.gameBoard.getDetectivePawns())
            elif (actionPawn.value in [ 5, 6, 7]):
                self.cardsRecognitionHelper.ComputeFrame(img)
            elif (actionPawn == ActionPawns.APAlibi):
                if IATurn == False:
                    self.gameBoard.get_alibi_card()
                    self.showAlibi = True
                    return
                else:
                    self.cardsRecognitionHelper.ComputeFrame(img)
        else:
            # In the other case, we check the cards to verify if alibi has been correctly removed
            self.cardsRecognitionHelper.ComputeFrame(img)

        # We then check if the action pawns has been respected
        if (self.gameBoard.IsActionPawnRespected(actionPawn.name)):
            if (actionPawn.value in [0,2,3,4]):
                self.gameBoard.updatePreviousPawnsState()
            elif (actionPawn.value in [1, 5, 6, 7]):
                self.gameBoard.updatePreviousCards()
            elif (actionPawn == ActionPawns.APAlibi and IATurn == False):
                # In case the user picks an alibi card we need to show a special state before validating the turn
                self.showAlibi = True
                return

            # We used the Action Pawns and now we remove it
            print("Action Pawn Used")
            self.pawnsRecognitionHelper.actionPawnUsed(actionPawn)
            self.gameBoard.nextTurn()

            # If we used all the action pawns move to the next Game Event : Manhunt
            if (len(self.gameBoard.getActionPawns()) == 0):
                print("Turn Finished")
                if (self.gameBoard.tryUpdateGameStatus(GameStates.GSAppealOfWitness)):
                    self.isJackSeen = self.cardsRecognitionHelper.IsInLineOfSight(img)
                    self.gameBoard.appealOfWitnesses(self.isJackSeen)
                    self.gameBoard.manhunt()
        else:
            print("Action Pawn not Validated")

    def ComputeMouseInput(self,event,x,y,flags,params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # If we are still at the homography stage record the first 4 points coordinates
            if len(self.list_board_coords) < 4:
                self.list_board_coords.append([x, y])
            else:
                actionPawnIndex = self.pawnsRecognitionHelper.actionPawnClick([x,y])

                # If we click on a (remaining) action pawn bounding box
                if actionPawnIndex is not None:
                    actionPawnClicked = self.gameBoard.getActionPawns()[actionPawnIndex]
                    print("Action Pawn Clicked : " + actionPawnClicked)

                    self.actionPawnClicked = ActionPawns[actionPawnClicked]